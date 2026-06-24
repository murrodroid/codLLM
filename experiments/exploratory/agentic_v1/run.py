"""Headless orchestrator: classify N CoD records via parallel `claude -p` calls.

Each record is classified by a fresh, isolated `claude` subprocess. The agent
sees only the CoD string — never the gold label — and must return its answer
as a JSON code block at the end of its response.

Usage:
    python -m experiments.agentic_baseline_claudecode.run --n 5
    python -m experiments.agentic_baseline_claudecode.run --n 500 --workers 5
    python -m experiments.agentic_baseline_claudecode.run --fresh   # ignore checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("claudecode_baseline")


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class RecordResult:
    cod: str
    gold: str
    parquet_idx: int = -1  # row id in the processed parquet, for held-out tracking
    predicted: str | None = None
    correct: bool = False
    reasoning: str = ""
    confidence: str = ""
    alternatives: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    num_turns: int = 0
    error: str | None = None
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Test sample
# ---------------------------------------------------------------------------


def load_eval_batch(start: int, n: int, seed: int) -> list[tuple[int, str, str]]:
    """Return [(parquet_idx, cod_text, gold_label), ...] from a stable shuffle of the test split.

    Uses sample(frac=1.0, random_state=seed) once to define a deterministic order,
    then slices [start:start+n]. Different (start, n) pairs with the same seed are
    guaranteed non-overlapping when start_a + n_a <= start_b. Verified to match
    sample(n=n, random_state=seed) for start=0 (so existing run_100 indices are
    recoverable).
    """
    from codllm.settings.schema import Config
    from codllm.data.handler import DataHandler

    cfg = Config()
    cfg.max_label_count = 1
    cfg.dataset_size = 1.0
    cfg.train_size = 0.9
    cfg.val_size = 0.05
    cfg.test_size = 0.05
    cfg.training_input = ["cod"]

    handler = DataHandler(cfg)
    full_df = handler.ensure_processed()
    splits = handler.split_dataframe(full_df)
    test_df = splits.test
    text_col = cfg.dataset_text_column
    label_col = cfg.dataset_label_column
    test_df = test_df[~test_df[label_col].str.contains(",", na=False)].copy()

    shuffled = test_df.sample(frac=1.0, random_state=seed)
    batch = shuffled.iloc[start : start + n]

    def extract_cod(text: str) -> str:
        parts = str(text).split(" | ")
        cod_part = parts[0]
        if cod_part.startswith("cod: "):
            cod_part = cod_part[5:]
        return cod_part.strip()

    return [
        (int(idx), extract_cod(row[text_col]), str(row[label_col]))
        for idx, row in batch.iterrows()
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
ICD10H_RE = re.compile(r"^[A-Z]\d{2}\.\d{3}$")


def _parse_response_json(text: str) -> dict[str, Any] | None:
    """Pull the agent's structured answer out of its free-form response.

    Strategy: prefer the last fenced JSON block; otherwise look for any object
    literal containing a "code" field.
    """
    matches = list(JSON_BLOCK_RE.finditer(text))
    if matches:
        for m in reversed(matches):
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text):
        candidate = m.group()
        if '"code"' in candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Single-record classification via `claude -p`
# ---------------------------------------------------------------------------


def _build_prompt(template: str, cod_text: str) -> str:
    return f'{template}\n\n---\n\nCause of death to classify: "{cod_text}"'


def classify_one(
    cod: str,
    gold: str,
    parquet_idx: int,
    prompt_template: str,
    model: str,
    timeout_s: int,
    claude_bin: str,
) -> RecordResult:
    result = RecordResult(cod=cod, gold=gold, parquet_idx=parquet_idx)
    prompt = _build_prompt(prompt_template, cod)

    # Prompt is piped via stdin to avoid the Windows .cmd-shim mangling argv
    # special characters (backticks, newlines) when the prompt is large.
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
        result.error = f"claude exit {proc.returncode}: {proc.stderr.strip()[:500]}"
        return result

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        result.error = f"failed to parse json envelope: {e}"
        result.raw_response = proc.stdout[:2000]
        return result

    if envelope.get("is_error"):
        result.error = f"agent reported error: {envelope.get('result', '')[:500]}"
        return result

    response_text = envelope.get("result", "") or ""
    result.raw_response = response_text
    result.cost_usd = float(envelope.get("total_cost_usd") or 0.0)
    result.num_turns = int(envelope.get("num_turns") or 0)

    parsed = _parse_response_json(response_text)
    if not parsed:
        result.error = "no parseable JSON block in response"
        return result

    code = str(parsed.get("code", "")).strip()
    if not ICD10H_RE.match(code):
        result.error = f"predicted code does not match X00.000 format: {code!r}"
        result.predicted = code or None
        return result

    result.predicted = code
    result.correct = code == gold
    result.reasoning = str(parsed.get("reasoning", ""))
    result.confidence = str(parsed.get("confidence", ""))
    alts = parsed.get("alternatives") or []
    result.alternatives = [str(a) for a in alts if isinstance(a, (str,))]
    return result


# ---------------------------------------------------------------------------
# Metrics + IO
# ---------------------------------------------------------------------------


def compute_metrics(results: list[RecordResult]) -> dict[str, Any]:
    """Compute the same metrics codllm.metrics emits, so the agentic and
    fine-tuned baselines can be compared head-to-head on the same data."""
    from codllm.metrics import _macro_precision_recall_f1

    n = len(results)
    valid = [r for r in results if r.predicted is not None]
    exact = sum(1 for r in valid if r.correct)
    no_answer = sum(1 for r in results if r.predicted is None)
    chapter = sum(1 for r in valid if r.predicted[0] == r.gold[0])
    category = sum(1 for r in valid if r.predicted[:5] == r.gold[:5])
    elapsed = sum(r.elapsed_s for r in results)
    cost = sum(r.cost_usd for r in results)

    pred_sets = [({r.predicted} if r.predicted else set()) for r in results]
    gold_sets = [{r.gold} for r in results]
    macro = _macro_precision_recall_f1(pred_sets, gold_sets)
    unique_classes = len(set().union(*pred_sets, *gold_sets))

    metrics: dict[str, Any] = {
        "n": n,
        "accuracy": exact / max(n, 1),  # micro F1 in the single-label regime
        "exact_match": exact,
        "no_answer": no_answer,
        "valid_predictions": len(valid),
        "chapter_accuracy": chapter / max(len(valid), 1),
        "category_accuracy": category / max(len(valid), 1),
        "unique_classes": unique_classes,
        "avg_elapsed_s": elapsed / max(n, 1),
        "total_elapsed_s": elapsed,
        "total_cost_usd": cost,
        "avg_cost_usd": cost / max(n, 1),
    }
    metrics.update(macro)  # macro_precision, macro_recall, macro_f1
    return metrics


def save_checkpoint(results: list[RecordResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    tmp.replace(path)


def load_checkpoint(path: Path) -> list[RecordResult]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_results(results: list[RecordResult], metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "predictions": [
            {
                "parquet_idx": r.parquet_idx,
                "cod": r.cod,
                "gold": r.gold,
                "predicted": r.predicted,
                "correct": r.correct,
                "confidence": r.confidence,
                "alternatives": r.alternatives,
                "reasoning": r.reasoning,
                "num_turns": r.num_turns,
                "elapsed_s": round(r.elapsed_s, 2),
                "cost_usd": round(r.cost_usd, 4),
                "error": r.error,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_held_out_index(
    results_dir: Path, batch_indices: list[int], batch_label: str
) -> Path:
    """Append a batch's parquet indices to the cumulative held-out file.

    Format: {"all_indices": [...], "batches": [{"label": str, "indices": [...]}, ...]}
    """
    path = results_dir / "held_out_indices.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    else:
        existing = {"all_indices": [], "batches": []}

    if any(b["label"] == batch_label for b in existing["batches"]):
        # Idempotent: replace the entry for this batch label.
        existing["batches"] = [b for b in existing["batches"] if b["label"] != batch_label]

    existing["batches"].append({"label": batch_label, "indices": sorted(batch_indices)})
    all_idx = sorted({i for b in existing["batches"] for i in b["indices"]})
    existing["all_indices"] = all_idx
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--start", type=int, default=0,
                   help="Slice offset into the deterministic shuffle (for non-overlapping batches)")
    p.add_argument("--batch-label", default=None,
                   help="Identifier for this batch in held_out_indices.json (defaults to start-end range)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--model", default="opus")
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-path", default=str(THIS_DIR / "results" / "results.json"))
    p.add_argument("--checkpoint-path", default=str(THIS_DIR / "results" / "checkpoint.pkl"))
    p.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint")
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
    results_path = Path(args.results_path)
    checkpoint_path = Path(args.checkpoint_path)

    batch_label = args.batch_label or f"start{args.start}_n{args.n}_seed{args.seed}"
    logger.info(
        "Loading test records: start=%d n=%d seed=%d (label=%s)",
        args.start, args.n, args.seed, batch_label,
    )
    records = load_eval_batch(args.start, args.n, args.seed)
    logger.info("Loaded %d records.", len(records))

    # Persist the parquet indices for this batch immediately, so they are
    # excludable from training even if the run is interrupted.
    update_held_out_index(
        results_path.parent, [idx for idx, _, _ in records], batch_label
    )

    completed: list[RecordResult] = []
    if not args.fresh and checkpoint_path.exists():
        completed = load_checkpoint(checkpoint_path)
        logger.info("Resuming from checkpoint: %d already done.", len(completed))

    done_indices = {r.parquet_idx for r in completed}
    pending = [t for t in records if t[0] not in done_indices]
    logger.info("Pending: %d records.", len(pending))

    if not pending:
        metrics = compute_metrics(completed)
        save_results(completed, metrics, results_path)
        print(json.dumps(metrics, indent=2))
        return 0

    t_start = time.time()
    finished_since_save = 0
    save_every = 5

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    classify_one,
                    cod,
                    gold,
                    parquet_idx,
                    prompt_template,
                    args.model,
                    args.timeout_s,
                    claude_bin,
                ): (parquet_idx, cod, gold)
                for parquet_idx, cod, gold in pending
            }
            for fut in as_completed(futures):
                parquet_idx, cod, gold = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # subprocess oddities — keep going
                    res = RecordResult(
                        cod=cod, gold=gold, parquet_idx=parquet_idx,
                        error=f"executor error: {exc}",
                    )
                completed.append(res)
                tag = "OK" if res.correct else ("MISS" if res.predicted else "NO_ANSWER")
                logger.info(
                    "[%d/%d] %s pred=%s gold=%s elapsed=%.1fs cost=$%.4f",
                    len(completed),
                    len(records),
                    tag,
                    res.predicted,
                    res.gold,
                    res.elapsed_s,
                    res.cost_usd,
                )
                finished_since_save += 1
                if finished_since_save >= save_every:
                    save_checkpoint(completed, checkpoint_path)
                    finished_since_save = 0
    except KeyboardInterrupt:
        logger.warning("Interrupted; saving checkpoint before exit.")
        save_checkpoint(completed, checkpoint_path)
        return 130

    save_checkpoint(completed, checkpoint_path)

    metrics = compute_metrics(completed)
    metrics["wallclock_s"] = round(time.time() - t_start, 1)
    save_results(completed, metrics, results_path)

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    print(json.dumps(metrics, indent=2))
    logger.info("Saved: %s", results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
