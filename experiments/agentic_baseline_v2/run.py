"""Agentic baseline v2: multi-label ICD-10h classification via isolated `claude -p` calls.

Each test record is classified by a fresh, isolated `claude` subprocess. The agent
sees only the CoD string -- never the gold label -- and must emit 1-3 ICD-10h
codes as a JSON code block. Every record is logged to a per-record JSONL stream
for crash-resilient post-hoc analysis.

Differences from the v1 baseline (experiments/agentic_baseline_claudecode/):
  - Multi-label support (max_label_count=3, no comma filter on test split)
  - Test split is the exact size_sweep split (data_seed=333, n=76,418), so
    macro_f1/sample_f1 are directly comparable to flan-t5 numbers
  - JSONL append per record + JSON checkpoint snapshot every K records
  - Multi-label parse (`codes` array, with fallback to legacy `code` singular)
  - Per-record metrics (precision/recall/F1 against gold set + exact-set match)
  - Final metrics use codllm.metrics primitives for bit-identical comparability
  - Logs source_id per record so per-source breakdowns are reconstructible

Usage:
    python -m experiments.agentic_baseline_v2.run --n 5             # smoke
    python -m experiments.agentic_baseline_v2.run --n 1000          # full run
    python -m experiments.agentic_baseline_v2.run --n 1000 --workers 4
    python -m experiments.agentic_baseline_v2.run --resume          # continue prior batch
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
PROMPT_PATH = THIS_DIR / "prompt.md"
RESULTS_DIR = THIS_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("agentic_baseline_v2")


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class RecordResult:
    parquet_idx: int
    source_id: str
    cod: str
    gold_str: str
    gold_codes: list[str]

    predicted_codes: list[str] = field(default_factory=list)
    predicted_codes_invalid: list[str] = field(default_factory=list)
    raw_codes_field: list[str] = field(default_factory=list)

    exact_set_match: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    reasoning: str = ""
    confidence: str = ""
    alternatives: list[str] = field(default_factory=list)

    num_turns: int = 0
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    raw_response: str = ""
    # Resolved model snapshot reported by the CLI envelope (e.g. claude-sonnet-4-5-...).
    # The --model arg may be an alias ("sonnet"); this records what it resolved to.
    model_used: str = ""
    # Set True by classify_one when the subprocess looks like the silent-quota
    # signature (returncode=1, empty stderr, no envelope in stdout, cost=0,
    # 2s <= elapsed <= 60s). Used by the consumer-loop streak guard to stop
    # the run cleanly when Sonnet weekly cap is hit and the CLI returns no
    # human-readable error to match on. Persisted to JSONL for forensics.
    quota_signal: bool = False

    def to_log_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # Truncate the raw_response in the per-record JSONL to keep file sane;
        # full raw is kept in the summary JSON only via results.json roll-up.
        if isinstance(d.get("raw_response"), str) and len(d["raw_response"]) > 8000:
            d["raw_response"] = d["raw_response"][:8000] + "...[truncated]"
        return d


# ---------------------------------------------------------------------------
# Test sample loading
# ---------------------------------------------------------------------------


def load_eval_batch(n: int, seed: int, index_file: str | None = None, input_mode: str = "metadata") -> list[tuple[int, str, str, str]]:
    """Return [(parquet_idx, source_id, cod_text, gold_label), ...] sampled from
    the size_sweep test split.

    Configures DataHandler with the EXACT same env the size_sweep flan-t5 runs
    use (max_label_count=3, default data_seed=333, dataset_size=1.0). Does NOT
    drop multi-label rows -- the agent must handle them.
    """
    from codllm.settings.schema import Config
    from codllm.data.handler import DataHandler

    cfg = Config()
    cfg.max_label_count = 3
    cfg.dataset_size = 1.0
    cfg.train_size = 0.9
    cfg.val_size = 0.05
    cfg.test_size = 0.05
    cfg.training_input = ["cod"] if input_mode == "cod" else ["cod", "age", "sex"]

    handler = DataHandler(cfg)
    full_df = handler.ensure_processed()
    splits = handler.split_dataframe(full_df)
    test_df = splits.test
    text_col = cfg.dataset_text_column
    label_col = cfg.dataset_label_column

    logger.info("Test split rows: %d (no comma filter applied)", len(test_df))

    if index_file:
        wanted = json.loads(Path(index_file).read_text(encoding="utf-8"))
        present = [int(i) for i in wanted if int(i) in test_df.index]
        logger.info("Index file %s: %d requested, %d present in test split", index_file, len(wanted), len(present))
        batch = test_df.loc[present]
    else:
        shuffled = test_df.sample(frac=1.0, random_state=seed)
        batch = shuffled.iloc[:n]

    src_col = "source_id" if "source_id" in batch.columns else None
    out: list[tuple[int, str, str, str]] = []
    for idx, row in batch.iterrows():
        # Pass the full processed input (cod + age + sex) so the agent sees the
        # same fields the fine-tuned models trained on; _build_prompt parses it.
        record_text = str(row[text_col]).strip()
        gold = str(row[label_col])
        src = str(row[src_col]) if src_col else ""
        out.append((int(idx), src, record_text, gold))
    return out


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
ICD10H_RE = re.compile(r"^[A-Z]\d{2}\.\d{3}$")


# Substrings that signal Sonnet-quota exhaustion or hard rate-limit -- in any of
# these cases the run is stopped immediately to avoid (a) burning further records
# on guaranteed failure and (b) any chance of the CLI silently switching models.
QUOTA_PATTERNS: tuple[str, ...] = (
    "usage limit reached",        # claude code subscription weekly / 5h cap
    "weekly usage limit",
    "5-hour limit",
    "monthly usage limit",
    "claude usage limit",
    "rate limit",
    "rate_limit_error",
    "credit balance is too low",
    "insufficient credit",
    "insufficient_quota",
    "billing_hard_limit_reached",
    "anthropic-rate-limit",
    "429",
)


def _is_quota_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    lowered = error_text.lower()
    return any(p in lowered for p in QUOTA_PATTERNS)


def _parse_response_json(text: str) -> dict[str, Any] | None:
    """Pull the agent's structured answer. Prefer the LAST fenced JSON block; on
    failure, look for any object literal containing "codes" or "code"."""
    matches = list(JSON_BLOCK_RE.finditer(text))
    for m in reversed(matches):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text):
        candidate = m.group()
        if '"codes"' in candidate or '"code"' in candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _coerce_codes(parsed: dict[str, Any]) -> list[str]:
    """Extract a codes list from either {"codes": [...]} or legacy {"code": "..."}.

    Accepts up to 3 entries; trims whitespace; drops empties. Validation against
    the ICD10H_RE happens in the caller so we can record the invalid ones."""
    raw = parsed.get("codes")
    if isinstance(raw, list) and raw:
        return [str(c).strip() for c in raw if str(c).strip()][:3]
    legacy = parsed.get("code")
    if isinstance(legacy, str) and legacy.strip():
        return [legacy.strip()]
    return []


def _split_gold(gold_str: str) -> list[str]:
    return [c.strip() for c in gold_str.split(",") if c.strip()]


def _per_record_prf1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Single-record classification via `claude -p`
# ---------------------------------------------------------------------------


def _parse_record_fields(text: str) -> tuple[str, str, str]:
    """Split a processed input (`cod: X | age: Y | sex: Z`) into (cod, age, sex)."""
    parts = str(text).split(" | ")
    cod = parts[0]
    if cod.startswith("cod: "):
        cod = cod[5:]
    fields: dict[str, str] = {}
    for p in parts[1:]:
        if ": " in p:
            k, v = p.split(": ", 1)
            fields[k.strip().lower()] = v.strip()
    return cod.strip(), fields.get("age", ""), fields.get("sex", "")


def _build_prompt(template: str, record_text: str) -> str:
    cod, age, sex = _parse_record_fields(record_text)
    lines = [f'Cause of death to classify: "{cod}"']
    if age and age.lower() not in ("", "unknown", "nan", "none"):
        lines.append(f"Recorded age: {age}")
    if sex and sex.lower() not in ("", "unknown", "nan", "none"):
        lines.append(f"Recorded sex: {sex}")
    return f"{template}\n\n---\n\n" + "\n".join(lines)


def classify_one(
    parquet_idx: int,
    source_id: str,
    cod: str,
    gold_str: str,
    prompt_template: str,
    model: str,
    timeout_s: int,
    claude_bin: str,
) -> RecordResult:
    gold_codes = _split_gold(gold_str)
    result = RecordResult(
        parquet_idx=parquet_idx,
        source_id=source_id,
        cod=cod,
        gold_str=gold_str,
        gold_codes=gold_codes,
    )
    prompt = _build_prompt(prompt_template, cod)

    cmd = [
        claude_bin,
        "-p",
        "--output-format", "json",
        "--model", model,
        "--no-session-persistence",
        "--permission-mode", "bypassPermissions",
        "--allowedTools", "Bash,Read,Grep,Glob,WebSearch,WebFetch",
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        result.error = f"timeout after {timeout_s}s"
        result.elapsed_s = float(timeout_s)
        return result

    result.elapsed_s = time.time() - t0

    if proc.returncode != 0:
        stderr_clean = (proc.stderr or "").strip()
        stdout_clean = (proc.stdout or "").strip()
        result.error = f"claude exit {proc.returncode}: {stderr_clean[:500]}"
        # Silent-quota signature: returncode=1, empty stderr, no parseable
        # envelope in stdout, elapsed <= 60s. The cost_usd check is implicit
        # because we have not yet set it from the envelope.
        # NOTE: the old 2.0s lower bound was dropped. The silent quota-exit
        # returns almost instantly (< 2s), so the floor let a whole window's
        # tail (202 records on the matched run, 2026-06-15) burn unflagged.
        # We keep only the 60s ceiling to exclude long real failures.
        if (
            proc.returncode == 1
            and not stderr_clean
            and result.elapsed_s <= 60.0
            and ('"result"' not in stdout_clean and '"is_error"' not in stdout_clean)
        ):
            result.quota_signal = True
        return result

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        result.error = f"failed to parse json envelope: {e}"
        result.raw_response = proc.stdout[:2000]
        return result

    if envelope.get("is_error"):
        result.error = f"agent reported error: {str(envelope.get('result', ''))[:500]}"
        return result

    response_text = envelope.get("result", "") or ""
    result.raw_response = response_text
    result.cost_usd = float(envelope.get("total_cost_usd") or 0.0)
    result.num_turns = int(envelope.get("num_turns") or 0)
    result.model_used = str(envelope.get("model") or "")
    if not result.model_used:
        _mu = envelope.get("modelUsage")
        if isinstance(_mu, dict) and _mu:
            result.model_used = ",".join(sorted(_mu.keys()))

    parsed = _parse_response_json(response_text)
    if not parsed:
        result.error = "no parseable JSON block in response"
        return result

    raw_codes = _coerce_codes(parsed)
    result.raw_codes_field = list(raw_codes)
    valid: list[str] = []
    invalid: list[str] = []
    for c in raw_codes:
        if ICD10H_RE.match(c):
            valid.append(c)
        else:
            invalid.append(c)
    result.predicted_codes = valid
    result.predicted_codes_invalid = invalid

    result.reasoning = str(parsed.get("reasoning", ""))
    result.confidence = str(parsed.get("confidence", ""))
    alts = parsed.get("alternatives") or []
    result.alternatives = [str(a) for a in alts if isinstance(a, str)]

    pred_set = set(valid)
    gold_set = set(gold_codes)
    result.exact_set_match = pred_set == gold_set and bool(pred_set)
    p, r, f1 = _per_record_prf1(pred_set, gold_set)
    result.precision = p
    result.recall = r
    result.f1 = f1

    if not valid:
        result.error = result.error or "no valid ICD-10h codes in response"
    return result


# ---------------------------------------------------------------------------
# Metrics + IO
# ---------------------------------------------------------------------------


def compute_metrics(results: list[RecordResult]) -> dict[str, Any]:
    """Compute multi-label metrics that mirror what codllm.metrics emits for the
    fine-tuned models, so head-to-head comparison is bit-identical."""
    from codllm.metrics import (
        _macro_precision_recall_f1,
        _micro_precision_recall_f1,
        _sample_precision_recall_f1,
    )

    n = len(results)
    valid_pred = [r for r in results if r.predicted_codes]
    pred_sets = [set(r.predicted_codes) for r in results]
    gold_sets = [set(r.gold_codes) for r in results]

    exact = sum(1 for r in results if r.exact_set_match)
    no_answer = sum(1 for r in results if not r.predicted_codes)
    invalid_records = sum(1 for r in results if r.predicted_codes_invalid)
    invalid_codes_total = sum(len(r.predicted_codes_invalid) for r in results)
    errors = sum(1 for r in results if r.error)

    macro = _macro_precision_recall_f1(pred_sets, gold_sets)
    micro = _micro_precision_recall_f1(pred_sets, gold_sets)
    sample = _sample_precision_recall_f1(pred_sets, gold_sets)

    elapsed = sum(r.elapsed_s for r in results)
    cost = sum(r.cost_usd for r in results)

    # Per-source breakdown
    by_source: dict[str, list[RecordResult]] = {}
    for r in results:
        by_source.setdefault(r.source_id, []).append(r)
    per_source_metrics: dict[str, dict[str, float]] = {}
    for src, rs in by_source.items():
        ps = [set(r.predicted_codes) for r in rs]
        gs = [set(r.gold_codes) for r in rs]
        m_src = _macro_precision_recall_f1(ps, gs)
        sm_src = _sample_precision_recall_f1(ps, gs)
        per_source_metrics[src] = {
            "n": len(rs),
            "exact_match_rate": sum(1 for r in rs if r.exact_set_match) / max(len(rs), 1),
            "macro_f1": m_src.get("macro_f1", 0.0),
            "sample_f1": sm_src.get("sample_f1", 0.0),
        }

    unique_classes = len(set().union(*pred_sets, *gold_sets))

    out: dict[str, Any] = {
        "n": n,
        "exact_set_match_rate": exact / max(n, 1),
        "exact_set_match": exact,
        "no_answer": no_answer,
        "errors": errors,
        "invalid_code_records": invalid_records,
        "invalid_codes_total": invalid_codes_total,
        "invalid_code_record_rate": invalid_records / max(n, 1),
        "valid_predictions": len(valid_pred),
        "unique_classes_seen": unique_classes,
        "avg_elapsed_s": elapsed / max(n, 1),
        "total_elapsed_s": elapsed,
        "avg_cost_usd": cost / max(n, 1),
        "total_cost_usd": cost,
        "per_source": per_source_metrics,
    }
    out.update(macro)
    out.update(micro)
    out.update(sample)
    return out


def append_jsonl(result: RecordResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_log_dict(), ensure_ascii=False) + "\n")


def load_jsonl_predictions(path: Path) -> list[RecordResult]:
    """Rehydrate completed records from a JSONL log for resume support."""
    if not path.exists():
        return []
    out: list[RecordResult] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                out.append(RecordResult(
                    parquet_idx=int(d["parquet_idx"]),
                    source_id=str(d.get("source_id", "")),
                    cod=str(d.get("cod", "")),
                    gold_str=str(d.get("gold_str", "")),
                    gold_codes=list(d.get("gold_codes", []) or []),
                    predicted_codes=list(d.get("predicted_codes", []) or []),
                    predicted_codes_invalid=list(d.get("predicted_codes_invalid", []) or []),
                    raw_codes_field=list(d.get("raw_codes_field", []) or []),
                    exact_set_match=bool(d.get("exact_set_match", False)),
                    precision=float(d.get("precision", 0.0) or 0.0),
                    recall=float(d.get("recall", 0.0) or 0.0),
                    f1=float(d.get("f1", 0.0) or 0.0),
                    reasoning=str(d.get("reasoning", "")),
                    confidence=str(d.get("confidence", "")),
                    alternatives=list(d.get("alternatives", []) or []),
                    num_turns=int(d.get("num_turns", 0) or 0),
                    elapsed_s=float(d.get("elapsed_s", 0.0) or 0.0),
                    cost_usd=float(d.get("cost_usd", 0.0) or 0.0),
                    error=d.get("error"),
                    raw_response=str(d.get("raw_response", "")),
                    quota_signal=bool(d.get("quota_signal", False)),
                ))
            except Exception:
                continue
    return out


def save_results_snapshot(results: list[RecordResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": compute_metrics(results),
        "predictions": [r.to_log_dict() for r in results],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5,
                   help="Number of test records to classify (random sample, default 5).")
    p.add_argument("--seed", type=int, default=333,
                   help="RNG seed for the test-split shuffle (default 333 = data_seed).")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of concurrent claude subprocesses (each record is isolated).")
    p.add_argument("--model", default="sonnet",
                   help="Claude model alias: opus|sonnet|haiku or full ID (default sonnet).")
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--batch-label", default=None,
                   help="Identifier for output filenames (default: n{N}_seed{SEED}_{model}).")
    p.add_argument("--index-file", default=None,
                   help="JSON list of parquet_idx to classify EXACTLY (overrides seed sampling). "
                        "Use to run a fixed, reference-excluded, matched record set across configs.")
    p.add_argument("--input-mode", default="metadata", choices=["cod", "metadata"],
                   help="cod = CoD text only; metadata = CoD + age + sex (default).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from any existing per-record JSONL log.")
    p.add_argument("--cost-cap-usd", type=float, default=None,
                   help="Abort if cumulative cost exceeds this many USD.")
    p.add_argument("--no-stop-on-quota", action="store_true",
                   help="Disable the automatic stop-on-quota guard. Default is to "
                        "stop the run cleanly when a Claude usage-limit or rate-limit "
                        "error is detected, so we never silently fall back to a different model.")
    p.add_argument("--quota-stop-threshold", type=int, default=20,
                   help="Stop after this many quota-shaped events in a row (default 20). "
                        "Must ALSO survive >= 3 minutes of wall-clock since the streak "
                        "started AND at least one prior successful record in the run. "
                        "The dual gate keeps a short local hiccup from killing the run.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    claude_bin = shutil.which("claude")
    if claude_bin is None:
        print("ERROR: 'claude' CLI not found on PATH.", file=sys.stderr)
        return 2

    if not PROMPT_PATH.exists():
        print(f"ERROR: prompt template not found at {PROMPT_PATH}.", file=sys.stderr)
        return 2

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    label = args.batch_label or f"n{args.n}_seed{args.seed}_{args.model}"
    predictions_jsonl = RESULTS_DIR / f"{label}.predictions.jsonl"
    snapshot_path = RESULTS_DIR / f"{label}.results.json"

    logger.info("Loading test records: n=%d seed=%d label=%s", args.n, args.seed, label)
    records = load_eval_batch(args.n, args.seed, index_file=args.index_file, input_mode=args.input_mode)
    logger.info("Loaded %d records to classify.", len(records))

    completed: list[RecordResult] = []
    if args.resume:
        completed = load_jsonl_predictions(predictions_jsonl)
        logger.info("Resume: %d records already in %s", len(completed), predictions_jsonl.name)

    done_indices = {r.parquet_idx for r in completed}
    pending = [r for r in records if r[0] not in done_indices]
    logger.info("Pending: %d / %d records.", len(pending), len(records))

    if not pending:
        save_results_snapshot(completed, snapshot_path)
        metrics = compute_metrics(completed)
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_source"}, indent=2))
        return 0

    cost_running = sum(r.cost_usd for r in completed)
    t_start = time.time()
    save_every = 5
    quota_streak = 0
    quota_streak_start_t: float | None = None
    successful_in_this_run = 0
    total_quota_signals_this_run = 0
    quota_exhausted = False
    fail_streak = 0
    QUOTA_STREAK_MIN_SECONDS = 180.0  # 3 minutes wall-clock since streak start
    # Cold-start backstop: if we accumulate this many quota-shaped events with
    # ZERO successes in the run, quota was already dead when the run started
    # (the dual-gate streak guard never arms because it requires a prior
    # success). Stop so an autonomous resume cannot silent-fail through all
    # remaining records.
    COLD_START_DEAD_QUOTA_LIMIT = 30
    # Classification-independent backstop. The silent quota-exit (returncode 1,
    # error envelope on stdout, empty stderr) leaves BOTH quota_signal and
    # _is_quota_error False and costs $0, so it hits neither the quota-shaped
    # streak nor the cost>0 reset -- the run burned its entire 2026-06-16 tail
    # (2086 records) this way. Any run of this many records with no usable
    # prediction means systematic failure (quota wall, rate limit, broken
    # setup); each costs $0, so the detection window is effectively free.
    FAIL_STREAK_LIMIT = 30

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    classify_one,
                    parquet_idx,
                    source_id,
                    cod,
                    gold_str,
                    prompt_template,
                    args.model,
                    args.timeout_s,
                    claude_bin,
                ): (parquet_idx, source_id, cod, gold_str)
                for parquet_idx, source_id, cod, gold_str in pending
            }
            for fut in as_completed(futures):
                parquet_idx, source_id, cod, gold_str = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = RecordResult(
                        parquet_idx=parquet_idx,
                        source_id=source_id,
                        cod=cod,
                        gold_str=gold_str,
                        gold_codes=_split_gold(gold_str),
                        error=f"executor error: {exc}",
                    )
                completed.append(res)
                append_jsonl(res, predictions_jsonl)
                cost_running += res.cost_usd

                tag = "OK" if res.exact_set_match else (
                    "PARTIAL" if res.f1 > 0 else (
                        "MISS" if res.predicted_codes else "NO_ANSWER"
                    )
                )
                logger.info(
                    "[%d/%d] %s src=%s pred=%s gold=%s f1=%.2f t=%.1fs $%.4f cum_cost=$%.2f",
                    len(completed),
                    len(records),
                    tag,
                    source_id[:18],
                    ",".join(res.predicted_codes) or "-",
                    res.gold_str,
                    res.f1,
                    res.elapsed_s,
                    res.cost_usd,
                    cost_running,
                )

                if len(completed) % save_every == 0:
                    save_results_snapshot(completed, snapshot_path)

                if args.cost_cap_usd is not None and cost_running > args.cost_cap_usd:
                    logger.error(
                        "Cost cap $%.2f exceeded (cum $%.2f) -- aborting.",
                        args.cost_cap_usd, cost_running,
                    )
                    break

                # Track whether THIS record proved the account is still answering.
                # Used to (a) arm the streak guard only after one prior success and
                # (b) reset the streak on any cost>0 record (real API response).
                if res.cost_usd > 0:
                    successful_in_this_run += 1

                # Consecutive-failure backstop (see FAIL_STREAK_LIMIT above):
                # increment on a FREE failure (errored, no usable prediction, and
                # $0 billed -- the quota/rate-limit signature), reset on anything
                # that produced codes OR cost money (a paid no-answer proves the
                # API is still serving, so it must not build the streak). Catches
                # the silent quota-exit the quota-shaped detector misses.
                if bool(res.error) and not res.predicted_codes and res.cost_usd == 0:
                    fail_streak += 1
                else:
                    fail_streak = 0
                if not args.no_stop_on_quota and fail_streak >= FAIL_STREAK_LIMIT:
                    logger.error(
                        "Hit consecutive-failure wall (%d records in a row with no usable "
                        "prediction; last error: %s). Stopping cleanly -- almost certainly "
                        "the quota wall the quota-shaped detector missed. Re-run with "
                        "--resume after the window resets.",
                        fail_streak, (res.error or "")[:160],
                    )
                    quota_exhausted = True
                    save_results_snapshot(completed, snapshot_path)
                    break

                if not args.no_stop_on_quota:
                    is_quota_shaped = _is_quota_error(res.error) or bool(res.quota_signal)
                    if is_quota_shaped:
                        total_quota_signals_this_run += 1
                        # Cold-start dead-quota backstop: many quota signals,
                        # zero successes ever -> quota was dead at launch.
                        if (
                            successful_in_this_run == 0
                            and total_quota_signals_this_run >= COLD_START_DEAD_QUOTA_LIMIT
                        ):
                            logger.error(
                                "Cold-start dead quota: %d quota-shaped events with 0 successes. "
                                "Quota was already exhausted at launch. Stopping cleanly; do NOT "
                                "auto-resume until the weekly limit resets.",
                                total_quota_signals_this_run,
                            )
                            quota_exhausted = True
                            save_results_snapshot(completed, snapshot_path)
                            break
                        if quota_streak == 0:
                            quota_streak_start_t = time.time()
                        quota_streak += 1
                        wallclock_in_streak = (
                            time.time() - quota_streak_start_t
                            if quota_streak_start_t is not None else 0.0
                        )
                        logger.warning(
                            "Quota-shaped event (streak %d/%d, %.0fs wall, prior_success=%d): %s",
                            quota_streak, args.quota_stop_threshold,
                            wallclock_in_streak, successful_in_this_run,
                            ("silent-fail" if res.quota_signal else (res.error or "")[:160]),
                        )
                        # Dual gate: count >= threshold AND >= 3 min wall-clock AND
                        # at least one prior success in THIS run. The wall-clock
                        # gate prevents a 30-second local hiccup at workers=4 from
                        # killing the run; the prior-success gate prevents
                        # cold-start auth bugs from aborting before a single record
                        # has proven the credentials work.
                        if (
                            quota_streak >= args.quota_stop_threshold
                            and wallclock_in_streak >= QUOTA_STREAK_MIN_SECONDS
                            and successful_in_this_run >= 1
                        ):
                            logger.error(
                                "Hit Sonnet quota wall (%d consecutive quota-shaped events over %.0fs of wall-clock; "
                                "%d prior successes confirm credentials work). Stopping run cleanly so we do not "
                                "silently fall back to a different model. Re-run with --resume after the weekly limit resets.",
                                quota_streak, wallclock_in_streak, successful_in_this_run,
                            )
                            quota_exhausted = True
                            save_results_snapshot(completed, snapshot_path)
                            break
                    elif res.cost_usd > 0:
                        # Real API response: reset the streak. Don't reset on a
                        # non-quota error (parse fail etc) that ALSO returned $0
                        # because that does not prove the API is still answering.
                        quota_streak = 0
                        quota_streak_start_t = None
    except KeyboardInterrupt:
        logger.warning("Interrupted; finalizing snapshot before exit.")
        save_results_snapshot(completed, snapshot_path)
        return 130

    save_results_snapshot(completed, snapshot_path)
    metrics = compute_metrics(completed)
    metrics["wallclock_s"] = round(time.time() - t_start, 1)
    metrics["stopped_via_quota"] = bool(quota_exhausted)
    save_results_snapshot(completed, snapshot_path)

    logger.info("=" * 60)
    if quota_exhausted:
        logger.info("RESULTS (%s) -- STOPPED VIA QUOTA AT n=%d", label, len(completed))
    else:
        logger.info("RESULTS (%s) -- COMPLETED %d records", label, len(completed))
    logger.info("=" * 60)
    headline = {k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
                if k not in ("per_source",)}
    print(json.dumps(headline, indent=2, default=str))
    logger.info("Per-record JSONL: %s", predictions_jsonl)
    logger.info("Snapshot:         %s", snapshot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
