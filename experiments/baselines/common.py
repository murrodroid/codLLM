"""Shared helpers for RQ1 baselines.

All baselines must evaluate on the exact same test split as the size_sweep
flan-t5 runs and the agentic baseline (data_seed=333, 90/5/5, max_label_count=3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RESULTS_DIR = REPO_ROOT / "experiments" / "baselines" / "results"
BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared per-record schema (mirrors the agentic baseline RecordResult)
# ---------------------------------------------------------------------------


@dataclass
class BaselineRecord:
    parquet_idx: int
    source_id: str
    cod: str
    gold_str: str
    gold_codes: list[str]
    predicted_codes: list[str] = field(default_factory=list)
    predicted_codes_invalid: list[str] = field(default_factory=list)
    exact_set_match: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Split loading (must match the size_sweep / agentic baseline config exactly)
# ---------------------------------------------------------------------------


def load_train_and_test(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) under the production size_sweep config.

    Uses data_seed=333, 90/5/5, max_label_count=3. Bit-identical splits to the
    agentic baseline and to flan-t5 size_sweep so RQ1 numbers are paired.
    """
    from codllm.settings.schema import Config
    from codllm.data.handler import DataHandler

    cfg = Config()
    cfg.max_label_count = 3
    cfg.dataset_size = 1.0
    cfg.train_size = 0.9
    cfg.val_size = 0.05
    cfg.test_size = 0.05
    import os
    if os.environ.get("CODLLM_BASELINE_FULL_TEXT", "").lower() in ("1", "true", "yes"):
        cfg.training_input = ["cod", "age", "sex"]
    else:
        cfg.training_input = ["cod"]

    handler = DataHandler(cfg)
    full = handler.ensure_processed()
    splits = handler.split_dataframe(full)
    if verbose:
        print(f"train: {len(splits.train):,}   val: {len(splits.val):,}   test: {len(splits.test):,}")
    return splits.train, splits.test


def extract_cod(text: str) -> str:
    """Pull the bare cause-of-death string out of the model-style input
    representation (`cod: <text> | ...`).

    If CODLLM_BASELINE_FULL_TEXT is set, return the full input (cod + age + sex)
    unchanged, so baselines can be run on the same input the fine-tuned models
    saw, for an apples-to-apples RQ1 comparison.
    """
    if not isinstance(text, str):
        return ""
    import os
    if os.environ.get("CODLLM_BASELINE_FULL_TEXT", "").lower() in ("1", "true", "yes"):
        return text.strip()
    parts = text.split(" | ")
    head = parts[0]
    if head.startswith("cod: "):
        head = head[5:]
    return head.strip()


# ---------------------------------------------------------------------------
# Per-record P/R/F1 (codllm.metrics has the macro/micro/sample aggregates)
# ---------------------------------------------------------------------------


def per_record_prf1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def split_gold(gold_str: str) -> list[str]:
    return [c.strip() for c in str(gold_str).split(",") if c.strip()]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def append_jsonl(rec: BaselineRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_log_dict(), ensure_ascii=False) + "\n")


def write_jsonl(records: list[BaselineRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_log_dict(), ensure_ascii=False) + "\n")
    import os
    os.replace(tmp, path)


def load_jsonl(path: Path) -> list[BaselineRecord]:
    out: list[BaselineRecord] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(BaselineRecord(
                parquet_idx=int(d["parquet_idx"]),
                source_id=str(d.get("source_id", "")),
                cod=str(d.get("cod", "")),
                gold_str=str(d.get("gold_str", "")),
                gold_codes=list(d.get("gold_codes", []) or []),
                predicted_codes=list(d.get("predicted_codes", []) or []),
                predicted_codes_invalid=list(d.get("predicted_codes_invalid", []) or []),
                exact_set_match=bool(d.get("exact_set_match", False)),
                precision=float(d.get("precision", 0.0) or 0.0),
                recall=float(d.get("recall", 0.0) or 0.0),
                f1=float(d.get("f1", 0.0) or 0.0),
                elapsed_s=float(d.get("elapsed_s", 0.0) or 0.0),
                cost_usd=float(d.get("cost_usd", 0.0) or 0.0),
                error=d.get("error"),
                extra=dict(d.get("extra", {}) or {}),
            ))
    return out


def compute_baseline_metrics(records: list[BaselineRecord]) -> dict[str, Any]:
    """Compute multi-label metrics that mirror the flan-t5 / agentic baseline."""
    from codllm.metrics import (
        _macro_precision_recall_f1,
        _micro_precision_recall_f1,
        _sample_precision_recall_f1,
    )
    n = len(records)
    pred_sets = [set(r.predicted_codes) for r in records]
    gold_sets = [set(r.gold_codes) for r in records]
    exact = sum(1 for r in records if r.exact_set_match)
    no_answer = sum(1 for r in records if not r.predicted_codes)
    errors = sum(1 for r in records if r.error)
    macro = _macro_precision_recall_f1(pred_sets, gold_sets)
    micro = _micro_precision_recall_f1(pred_sets, gold_sets)
    sample = _sample_precision_recall_f1(pred_sets, gold_sets)
    elapsed = sum(r.elapsed_s for r in records)
    cost = sum(r.cost_usd for r in records)

    by_source: dict[str, list[BaselineRecord]] = {}
    for r in records:
        by_source.setdefault(r.source_id, []).append(r)
    per_source: dict[str, dict[str, float]] = {}
    for src, rs in by_source.items():
        ps = [set(r.predicted_codes) for r in rs]
        gs = [set(r.gold_codes) for r in rs]
        per_source[src] = {
            "n": len(rs),
            "exact_match_rate": sum(1 for r in rs if r.exact_set_match) / max(len(rs), 1),
            "macro_f1": _macro_precision_recall_f1(ps, gs).get("macro_f1", 0.0),
            "sample_f1": _sample_precision_recall_f1(ps, gs).get("sample_f1", 0.0),
        }

    out: dict[str, Any] = {
        "n": n,
        "exact_set_match_rate": exact / max(n, 1),
        "exact_set_match": exact,
        "no_answer": no_answer,
        "errors": errors,
        "total_elapsed_s": elapsed,
        "avg_elapsed_s": elapsed / max(n, 1),
        "total_cost_usd": cost,
        "avg_cost_usd": cost / max(n, 1),
        "per_source": per_source,
    }
    out.update(macro)
    out.update(micro)
    out.update(sample)
    return out


def save_snapshot(records: list[BaselineRecord], path: Path) -> None:
    import os
    payload = {
        "metrics": compute_baseline_metrics(records),
        "predictions_file": str(path).replace(".results.json", ".predictions.jsonl"),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
