"""Calibration + risk-coverage analysis on uncertainty inference output.

Reads the JSONL produced by `experiments.uncertainty.inference` and computes:

1. **Calibration** for each scoring signal (mean_logprob etc):
   - Reliability diagram (binned accuracy vs binned confidence).
   - Expected Calibration Error (ECE).
   - Optimal temperature scaling factor (single-parameter post-hoc fit on a
     held-out calibration subset). After fitting, ECE is recomputed; the diff
     tells you whether the raw signal is calibratable.

2. **Risk-coverage curve** for each signal:
   - Sort predictions by descending confidence (= use the signal as a ranker).
   - Sweep coverage from 100% down to 0% by dropping the lowest-confidence
     predictions; record accuracy on the retained set.
   - AURC (area under the risk-coverage curve) is the scalar summary —
     lower is better (less risk for given coverage).
   - Coverage at a target precision (e.g., 95% accuracy on retained) is the
     directly product-relevant number for "what fraction can we auto-accept?"

Pure post-hoc — runs on the JSONL, no model required.

Usage:
    python -m experiments.uncertainty.analysis \\
        --input experiments/uncertainty/results/val_run-XXXX.jsonl \\
        --output-dir experiments/uncertainty/results/analysis-run-XXXX
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("uncertainty.analysis")


# Signals where higher = more confident. For entropy-style signals we negate
# them at scoring time so all signals follow the "higher = more confident" rule.
SIGNAL_DIRECTION = {
    "sum_logprob": +1,
    "mean_logprob": +1,
    "min_logprob": +1,
    "mean_entropy": -1,
    "first_token_entropy": -1,
}


def _confidence_from_signal(df: pd.DataFrame, signal: str) -> np.ndarray:
    """Return a confidence score (higher = more confident) for the named signal."""
    direction = SIGNAL_DIRECTION[signal]
    raw = df[signal].to_numpy()
    return raw * direction


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _logits_to_prob(score: np.ndarray) -> np.ndarray:
    """Map a real-valued confidence score into [0, 1] via sigmoid.

    Logprobs are already log-probs in (-inf, 0], so sigmoid gives a smooth
    monotone projection into [0, 1]. After temperature scaling this becomes
    a calibrated probability of correctness.
    """
    return 1.0 / (1.0 + np.exp(-score))


def _expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, *, n_bins: int = 15
) -> float:
    """Standard ECE: weighted abs gap between binned accuracy and binned conf."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _fit_temperature(
    score: np.ndarray, correct: np.ndarray, *, max_iter: int = 200
) -> float:
    """One-parameter logistic regression on (score / T) to fit a temperature.

    Solves min_T BCE(sigmoid(score / T), correct). Uses simple bisection on
    the gradient — keeps the dependency footprint at numpy only.
    """
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(correct, dtype=np.float64)

    def _bce_grad(T: float) -> float:
        T = max(T, 1e-6)
        z = score / T
        p = 1.0 / (1.0 + np.exp(-z))
        # d(BCE)/d(1/T) = mean((p - y) * score), so d/dT = -d(BCE)/d(1/T) / T^2
        return float(np.mean((p - target) * score) * (-1.0 / (T * T)))

    lo, hi = 0.05, 50.0
    glo, ghi = _bce_grad(lo), _bce_grad(hi)
    if glo * ghi > 0:
        # Gradient doesn't change sign in the search range; fall back to T=1.
        return 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gmid = _bce_grad(mid)
        if abs(gmid) < 1e-8 or (hi - lo) < 1e-5:
            return float(mid)
        if gmid * glo < 0:
            hi, ghi = mid, gmid
        else:
            lo, glo = mid, gmid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Risk-coverage curve / AURC
# ---------------------------------------------------------------------------


def _risk_coverage(
    confidences: np.ndarray, correct: np.ndarray
) -> pd.DataFrame:
    """Sweep coverage from 100% down to 0% and record retained accuracy."""
    order = np.argsort(-confidences)  # most confident first
    correct_sorted = correct[order].astype(int)
    cumulative_correct = np.cumsum(correct_sorted)
    coverages = np.arange(1, len(correct_sorted) + 1) / len(correct_sorted)
    accuracies = cumulative_correct / np.arange(1, len(correct_sorted) + 1)
    risks = 1.0 - accuracies
    return pd.DataFrame({
        "coverage": coverages,
        "accuracy": accuracies,
        "risk": risks,
    })


def _aurc(curve: pd.DataFrame) -> float:
    """Area under the risk-coverage curve (trapezoidal). Lower is better."""
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapezoid(curve["risk"].to_numpy(), curve["coverage"].to_numpy()))


def _coverage_at_target_accuracy(curve: pd.DataFrame, target: float) -> float:
    """Largest coverage at which retained accuracy >= target (0 if never)."""
    feasible = curve[curve["accuracy"] >= target]
    if feasible.empty:
        return 0.0
    return float(feasible["coverage"].max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class SignalReport:
    signal: str
    aurc: float
    cov_at_95: float
    cov_at_99: float
    ece_raw: float
    ece_calibrated: float
    temperature: float


def _analyze_one_signal(
    df: pd.DataFrame, signal: str, *, target_accs: list[float]
) -> tuple[SignalReport, pd.DataFrame]:
    confidences = _confidence_from_signal(df, signal)
    correct = df["correct"].to_numpy().astype(int)

    # Calibration: fit temperature on full set (post-hoc analysis only).
    temperature = _fit_temperature(confidences, correct)
    raw_probs = _logits_to_prob(confidences)
    cal_probs = _logits_to_prob(confidences / max(temperature, 1e-6))
    ece_raw = _expected_calibration_error(raw_probs, correct)
    ece_cal = _expected_calibration_error(cal_probs, correct)

    # Risk-coverage.
    rc = _risk_coverage(confidences, correct)
    aurc = _aurc(rc)
    cov_95 = _coverage_at_target_accuracy(rc, 0.95)
    cov_99 = _coverage_at_target_accuracy(rc, 0.99)

    report = SignalReport(
        signal=signal,
        aurc=aurc,
        cov_at_95=cov_95,
        cov_at_99=cov_99,
        ece_raw=ece_raw,
        ece_calibrated=ece_cal,
        temperature=temperature,
    )
    return report, rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSONL produced by inference.py")
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write per-signal CSVs and the summary JSON"
    )
    p.add_argument(
        "--target-accs", nargs="+", type=float, default=[0.95, 0.99],
        help="Target retained-accuracy thresholds for coverage reporting"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", in_path)
    df = pd.read_json(in_path, lines=True)
    logger.info("Loaded %d records, baseline accuracy=%.4f", len(df), df["correct"].mean())

    summary: list[dict[str, float]] = []
    for signal in SIGNAL_DIRECTION:
        if signal not in df.columns:
            logger.warning("Signal %s missing from input; skipping", signal)
            continue
        report, curve = _analyze_one_signal(df, signal, target_accs=args.target_accs)
        curve.to_csv(out_dir / f"rc_curve_{signal}.csv", index=False)
        summary.append({
            "signal": report.signal,
            "aurc": report.aurc,
            "cov_at_95": report.cov_at_95,
            "cov_at_99": report.cov_at_99,
            "ece_raw": report.ece_raw,
            "ece_calibrated": report.ece_calibrated,
            "temperature": report.temperature,
        })
        logger.info(
            "%s: AURC=%.4f  cov@0.95=%.3f  cov@0.99=%.3f  "
            "ECE=%.3f -> %.3f (T=%.3f)",
            report.signal,
            report.aurc, report.cov_at_95, report.cov_at_99,
            report.ece_raw, report.ece_calibrated, report.temperature,
        )

    summary_df = pd.DataFrame(summary).sort_values("aurc")
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Summary (sorted by AURC, lowest is best):\n%s", summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
