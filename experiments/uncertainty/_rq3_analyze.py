"""STEP 4: per-source + pooled risk-coverage metrics for the LOSO held-out runs.

Reuses the risk-coverage logic from experiments.uncertainty.analysis verbatim
(_risk_coverage, _aurc, _coverage_at_target_accuracy, _confidence_from_signal).
Ranks by mean_entropy (most-confident first). "correct" is the exact multi-label
set match already emitted by local_infer.py (pred_codes == gold_codes).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.uncertainty.analysis import (
    _aurc,
    _confidence_from_signal,
    _coverage_at_target_accuracy,
    _risk_coverage,
)

SRCS = ["amsterdam", "belgium", "copenhagen", "ipswich", "madrid"]
WANDB = {  # plan's W&B-logged held-out exact-match accuracy
    "amsterdam": 0.5028,
    "belgium": 0.5503,
    "copenhagen": 0.4252,
    "ipswich": 0.4097,
    "madrid": 0.5147,
}
SIGNAL = "mean_entropy"
BASE = Path("experiments/uncertainty/results/loso")


def _retained_acc_at_coverage(curve: pd.DataFrame, cov: float) -> float:
    """Retained accuracy when keeping the top `cov` fraction (most confident first)."""
    # curve coverage is increasing in 1/N steps; pick the row closest to (but not below) cov
    feasible = curve[curve["coverage"] <= cov + 1e-12]
    if feasible.empty:
        return float("nan")
    return float(feasible.iloc[-1]["accuracy"])


def analyze(df: pd.DataFrame) -> dict:
    conf = _confidence_from_signal(df, SIGNAL)
    correct = df["correct"].to_numpy().astype(int)
    curve = _risk_coverage(conf, correct)
    return {
        "n": int(len(df)),
        "accuracy": float(correct.mean()),
        "aurc": _aurc(curve),
        "retained_acc_at_80": _retained_acc_at_coverage(curve, 0.80),
        "retained_acc_at_90": _retained_acc_at_coverage(curve, 0.90),
        "cov_at_95": _coverage_at_target_accuracy(curve, 0.95),
        "cov_at_99": _coverage_at_target_accuracy(curve, 0.99),
    }


def main() -> int:
    per_source = {}
    frames = []
    for src in SRCS:
        path = BASE / f"{src}_holdout.jsonl"
        df = pd.read_json(path, lines=True)
        frames.append(df)
        per_source[src] = analyze(df)

    pooled_df = pd.concat(frames, ignore_index=True)
    pooled = analyze(pooled_df)

    # --- print deliverable ---
    print("=" * 110)
    print("PER-SOURCE held-out confidence-vs-accuracy (rank by mean_entropy, most-confident first)")
    print("=" * 110)
    hdr = f"{'source':11s} {'n':>6s} {'acc':>7s} {'wandb':>7s} {'delta':>7s} {'AURC':>7s} {'ret@80':>7s} {'ret@90':>7s} {'cov@95':>7s} {'cov@99':>7s}"
    print(hdr)
    for src in SRCS:
        m = per_source[src]
        wb = WANDB[src]
        d = m["accuracy"] - wb
        print(f"{src:11s} {m['n']:6d} {m['accuracy']:7.4f} {wb:7.4f} {d:+7.4f} "
              f"{m['aurc']:7.4f} {m['retained_acc_at_80']:7.4f} {m['retained_acc_at_90']:7.4f} "
              f"{m['cov_at_95']:7.4f} {m['cov_at_99']:7.4f}")
    print("-" * 110)
    print(f"{'POOLED':11s} {pooled['n']:6d} {pooled['accuracy']:7.4f} {'':7s} {'':7s} "
          f"{pooled['aurc']:7.4f} {pooled['retained_acc_at_80']:7.4f} {pooled['retained_acc_at_90']:7.4f} "
          f"{pooled['cov_at_95']:7.4f} {pooled['cov_at_99']:7.4f}")
    print("=" * 110)

    # --- W&B match verdict ---
    print("\nW&B MATCH VERDICT (local held-out exact-match acc vs W&B holdout/accuracy):")
    max_abs = 0.0
    for src in SRCS:
        d = per_source[src]["accuracy"] - WANDB[src]
        max_abs = max(max_abs, abs(d))
        verdict = "MATCH" if abs(d) < 0.005 else ("CLOSE" if abs(d) < 0.02 else "DIFFER")
        print(f"  {src:11s} local={per_source[src]['accuracy']:.4f}  wandb={WANDB[src]:.4f}  delta={d:+.4f}  {verdict}")
    print(f"  max |delta| across sources = {max_abs:.4f}")

    # --- pooled deferral headline ---
    print("\nPOOLED DEFERRAL HEADLINE:")
    print(f"  overall held-out accuracy (100% coverage) = {pooled['accuracy']:.4f}")
    print(f"  retained accuracy at 80% coverage         = {pooled['retained_acc_at_80']:.4f}")
    lift = pooled["retained_acc_at_80"] - pooled["accuracy"]
    print(f"  lift from deferring least-confident 20%    = {lift:+.4f}")

    out = BASE / "rq3_summary.json"
    out.write_text(json.dumps({"per_source": per_source, "pooled": pooled, "wandb": WANDB}, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
