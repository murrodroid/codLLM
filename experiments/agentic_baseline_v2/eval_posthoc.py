"""Post-hoc analysis for the agentic baseline v2 predictions.

Reads a predictions.jsonl file produced by run.py and emits:
  - Headline multi-label metrics (macro/micro/sample F1, exact-set rate)
  - Per-source breakdown table
  - Rare-codes slice (codes with <K training occurrences)
  - Invalid-code rate + error breakdown
  - Confidence calibration (predicted-confidence vs empirical accuracy)
  - Multi-label specific stats (single vs multi gold breakdown)
  - Sample of wrong predictions for spot-checking

Usage:
    python -m experiments.agentic_baseline_v2.eval_posthoc \
        results/n1000_seed333_sonnet.predictions.jsonl

    python -m experiments.agentic_baseline_v2.eval_posthoc \
        results/n1000_seed333_sonnet.predictions.jsonl \
        --rare-threshold 5 --save-table results/n1000_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]


def load_predictions(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _set(codes: Any) -> set[str]:
    if not codes:
        return set()
    if isinstance(codes, str):
        return {c.strip() for c in codes.split(",") if c.strip()}
    return {str(c).strip() for c in codes if str(c).strip()}


def _macro_f1(records: list[dict[str, Any]]) -> dict[str, float]:
    from codllm.metrics import (
        _macro_precision_recall_f1,
        _micro_precision_recall_f1,
        _sample_precision_recall_f1,
    )
    pred_sets = [_set(r.get("predicted_codes")) for r in records]
    gold_sets = [_set(r.get("gold_codes")) for r in records]
    out: dict[str, float] = {}
    out.update(_macro_precision_recall_f1(pred_sets, gold_sets))
    out.update(_micro_precision_recall_f1(pred_sets, gold_sets))
    out.update(_sample_precision_recall_f1(pred_sets, gold_sets))
    return out


def load_train_label_counts() -> dict[str, int]:
    """Count each code's occurrences in the size_sweep training split, so we
    can define a 'rare' slice in the test set."""
    from codllm.settings.schema import Config
    from codllm.data.handler import DataHandler

    cfg = Config()
    cfg.max_label_count = 3
    cfg.dataset_size = 1.0
    cfg.train_size = 0.9
    cfg.val_size = 0.05
    cfg.test_size = 0.05
    cfg.training_input = ["cod"]

    handler = DataHandler(cfg)
    full_df = handler.ensure_processed()
    splits = handler.split_dataframe(full_df)
    train_df = splits.train
    label_col = cfg.dataset_label_column

    counts: Counter[str] = Counter()
    for raw in train_df[label_col].astype(str):
        for c in (s.strip() for s in raw.split(",")):
            if c:
                counts[c] += 1
    return dict(counts)


def per_source_table(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_src[r.get("source_id", "")].append(r)
    out: dict[str, dict[str, Any]] = {}
    for src, rs in sorted(by_src.items()):
        exact = sum(1 for r in rs if r.get("exact_set_match"))
        m = _macro_f1(rs)
        out[src or "<unknown>"] = {
            "n": len(rs),
            "exact_set_match_rate": round(exact / max(len(rs), 1), 4),
            "macro_f1": round(m.get("macro_f1", 0.0), 4),
            "sample_f1": round(m.get("sample_f1", 0.0), 4),
            "micro_f1": round(m.get("micro_f1", 0.0), 4),
        }
    return out


def rare_slice(
    records: list[dict[str, Any]],
    train_counts: dict[str, int],
    threshold: int,
) -> dict[str, Any]:
    """A test record is 'rare' if AT LEAST ONE of its gold codes has fewer than
    `threshold` occurrences in train. Reports rate and macro/sample F1 on the slice."""
    rare = []
    for r in records:
        gold = _set(r.get("gold_codes"))
        if any(train_counts.get(c, 0) < threshold for c in gold):
            rare.append(r)
    m = _macro_f1(rare) if rare else {}
    return {
        "threshold": threshold,
        "n_rare": len(rare),
        "rare_share": round(len(rare) / max(len(records), 1), 4),
        "macro_f1": round(m.get("macro_f1", 0.0), 4) if rare else None,
        "sample_f1": round(m.get("sample_f1", 0.0), 4) if rare else None,
        "exact_set_match_rate": (
            round(sum(1 for r in rare if r.get("exact_set_match")) / max(len(rare), 1), 4)
            if rare else None
        ),
    }


def confidence_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Empirical accuracy by predicted-confidence bucket (high/medium/low/other)."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        c = (r.get("confidence") or "").strip().lower()
        if c not in ("high", "medium", "low"):
            c = "other"
        buckets[c].append(r)
    out: dict[str, Any] = {}
    for k in ("high", "medium", "low", "other"):
        rs = buckets.get(k, [])
        out[k] = {
            "n": len(rs),
            "exact_set_match_rate": round(
                sum(1 for r in rs if r.get("exact_set_match")) / max(len(rs), 1), 4
            ),
            "sample_f1_mean": round(
                sum(float(r.get("f1", 0.0)) for r in rs) / max(len(rs), 1), 4
            ),
        }
    return out


def multilabel_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Split by gold cardinality: how does the agent do on 1-code vs >=2-code gold?"""
    by_card: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        n = len(_set(r.get("gold_codes")))
        by_card[n].append(r)
    out: dict[str, Any] = {}
    for k in sorted(by_card):
        rs = by_card[k]
        m = _macro_f1(rs)
        out[f"gold_cardinality_{k}"] = {
            "n": len(rs),
            "exact_set_match_rate": round(
                sum(1 for r in rs if r.get("exact_set_match")) / max(len(rs), 1), 4
            ),
            "macro_f1": round(m.get("macro_f1", 0.0), 4),
            "sample_f1": round(m.get("sample_f1", 0.0), 4),
        }
    # Also: did the agent emit multi-code answers when gold was single?
    overpred = sum(
        1 for r in records
        if len(_set(r.get("gold_codes"))) == 1 and len(_set(r.get("predicted_codes"))) > 1
    )
    underpred = sum(
        1 for r in records
        if len(_set(r.get("gold_codes"))) > 1 and len(_set(r.get("predicted_codes"))) == 1
    )
    out["over_prediction_count"] = overpred
    out["under_prediction_count"] = underpred
    return out


def sample_failures(records: list[dict[str, Any]], k: int = 10) -> list[dict[str, Any]]:
    fails = [r for r in records if not r.get("exact_set_match") and not r.get("error")]
    sample = fails[:k]
    return [
        {
            "parquet_idx": r.get("parquet_idx"),
            "source_id": r.get("source_id"),
            "cod": r.get("cod"),
            "gold_codes": r.get("gold_codes"),
            "predicted_codes": r.get("predicted_codes"),
            "f1": r.get("f1"),
            "confidence": r.get("confidence"),
            "reasoning": (r.get("reasoning") or "")[:240],
        }
        for r in sample
    ]


def cost_and_latency(records: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(r.get("elapsed_s", 0.0)) for r in records]
    cost = [float(r.get("cost_usd", 0.0)) for r in records]
    return {
        "n": len(records),
        "total_cost_usd": round(sum(cost), 4),
        "avg_cost_usd": round(sum(cost) / max(len(cost), 1), 4),
        "total_elapsed_s": round(sum(elapsed), 1),
        "avg_elapsed_s": round(sum(elapsed) / max(len(elapsed), 1), 2),
        "max_elapsed_s": round(max(elapsed) if elapsed else 0.0, 1),
    }


def error_breakdown(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in records:
        e = r.get("error")
        if e:
            # Bucket by first few words
            head = " ".join(str(e).split()[:6])
            counts[head] += 1
    return dict(counts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions_jsonl", type=Path,
                   help="Path to per-record JSONL produced by run.py")
    p.add_argument("--rare-threshold", type=int, default=5,
                   help="A gold code is 'rare' if it appears <K times in train (default 5).")
    p.add_argument("--save-table", type=Path, default=None,
                   help="Write the full summary JSON to this path.")
    p.add_argument("--skip-rare", action="store_true",
                   help="Skip the rare-codes slice (avoids loading the train split).")
    p.add_argument("--n-fail-samples", type=int, default=10)
    args = p.parse_args()

    if not args.predictions_jsonl.exists():
        print(f"ERROR: {args.predictions_jsonl} not found", file=sys.stderr)
        return 2

    records = load_predictions(args.predictions_jsonl)
    if not records:
        print("ERROR: no records loaded.", file=sys.stderr)
        return 2

    headline = _macro_f1(records)
    headline["n"] = len(records)
    headline["exact_set_match_rate"] = round(
        sum(1 for r in records if r.get("exact_set_match")) / len(records), 4
    )
    headline["invalid_code_record_rate"] = round(
        sum(1 for r in records if r.get("predicted_codes_invalid")) / len(records), 4
    )
    headline["no_answer_rate"] = round(
        sum(1 for r in records if not r.get("predicted_codes")) / len(records), 4
    )
    for k in ("macro_f1", "micro_f1", "sample_f1",
              "macro_precision", "macro_recall",
              "micro_precision", "micro_recall",
              "sample_precision", "sample_recall"):
        if k in headline:
            headline[k] = round(headline[k], 4)

    rare = None
    if not args.skip_rare:
        try:
            tcounts = load_train_label_counts()
            rare = rare_slice(records, tcounts, threshold=args.rare_threshold)
        except Exception as exc:
            rare = {"error": f"failed to compute rare slice: {exc}"}

    summary = {
        "predictions_file": str(args.predictions_jsonl),
        "headline": headline,
        "per_source": per_source_table(records),
        "rare_slice": rare,
        "confidence_calibration": confidence_calibration(records),
        "multilabel_breakdown": multilabel_breakdown(records),
        "cost_and_latency": cost_and_latency(records),
        "error_breakdown": error_breakdown(records),
        "sample_failures": sample_failures(records, args.n_fail_samples),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    if args.save_table:
        args.save_table.parent.mkdir(parents=True, exist_ok=True)
        args.save_table.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\nFull summary written to {args.save_table}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
