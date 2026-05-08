"""Risk-coverage curves for selective prediction by quantile/coverage thresholds.

Coverage = fraction of records kept by accepting only the most-confident
predictions. For each coverage value c we keep the floor(c * n) records with
highest *confidence* (most-confident → lowest uncertainty for entropy-style
signals; most-confident → highest score for logprob-style signals), then
evaluate the chosen metric on the kept set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

UNCERTAINTY_SIGNAL_NAMES: tuple[str, ...] = (
    "sum_logprob",
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
    "first_token_entropy",
)

# +1 = higher value means more confident (logprobs).
# -1 = higher value means less confident (entropies). Sort accordingly.
UNCERTAINTY_SIGNAL_DIRECTION: dict[str, int] = {
    "sum_logprob": 1,
    "mean_logprob": 1,
    "min_logprob": 1,
    "mean_entropy": -1,
    "first_token_entropy": -1,
}

DEFAULT_COVERAGES: tuple[float, ...] = (
    1.0,
    0.95,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
)


@dataclass
class RcCurvePoint:
    """One row on a risk-coverage curve."""

    coverage: float
    threshold: float
    n_kept: int
    metric_value: float


def _kept_indices_for_coverage(
    confidences: Sequence[float],
    coverage: float,
) -> tuple[list[int], float]:
    """Return indices of the top-k most-confident records and the threshold value.

    Confidences must already be oriented so larger-is-better. ``coverage`` is
    clamped to (0, 1]; coverage<=0 returns an empty kept set with -inf threshold.
    """
    n = len(confidences)
    if n == 0:
        return [], float("-inf")
    if coverage >= 1.0:
        return list(range(n)), float("-inf")
    if coverage <= 0.0:
        return [], float("inf")

    keep_count = max(1, int(np.floor(coverage * n)))
    sorted_indices = sorted(
        range(n), key=lambda idx: confidences[idx], reverse=True
    )
    kept = sorted_indices[:keep_count]
    threshold = float(confidences[sorted_indices[keep_count - 1]])
    return kept, threshold


def _orient_as_confidence(
    raw_scores: Sequence[float],
    direction: int,
) -> list[float]:
    """Flip uncertainty scores so higher values consistently mean more confident."""
    if direction == 1:
        return [float(value) for value in raw_scores]
    if direction == -1:
        return [-float(value) for value in raw_scores]
    raise ValueError("direction must be +1 (logprob-style) or -1 (entropy-style).")


def coverage_at_threshold(
    confidences: Sequence[float],
    threshold: float,
) -> float:
    """Return the fraction of records with confidence >= threshold."""
    if not confidences:
        return 0.0
    kept = sum(1 for value in confidences if value >= threshold)
    return float(kept / len(confidences))


def compute_rc_curve_by_coverage(
    raw_scores: Sequence[float],
    metric_inputs: Sequence,
    *,
    metric_fn: Callable[[Sequence], float],
    direction: int,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> list[RcCurvePoint]:
    """Evaluate ``metric_fn`` on the kept subset for each coverage value.

    ``metric_inputs[i]`` is whatever ``metric_fn`` needs for record i (a bool
    for accuracy, a (predicted_set, label_set) pair for sample_f1, etc.).
    Returns one RcCurvePoint per requested coverage, sorted by coverage desc.
    """
    if len(raw_scores) != len(metric_inputs):
        raise ValueError("raw_scores and metric_inputs must have equal length.")

    confidences = _orient_as_confidence(raw_scores, direction)
    sorted_coverages = sorted({float(c) for c in coverages}, reverse=True)

    points: list[RcCurvePoint] = []
    for coverage in sorted_coverages:
        kept_indices, threshold = _kept_indices_for_coverage(confidences, coverage)
        if kept_indices:
            kept_inputs = [metric_inputs[idx] for idx in kept_indices]
            metric_value = float(metric_fn(kept_inputs))
        else:
            metric_value = float("nan")
        points.append(
            RcCurvePoint(
                coverage=float(coverage),
                threshold=float(threshold),
                n_kept=len(kept_indices),
                metric_value=metric_value,
            )
        )
    return points


def accuracy_metric(matches: Sequence[bool]) -> float:
    """Mean of boolean matches; 0.0 for empty input."""
    if not matches:
        return 0.0
    return float(sum(1 for value in matches if value) / len(matches))


def sample_f1_metric(pairs: Sequence[tuple[set[str], set[str]]]) -> float:
    """Average per-sample F1 over (predicted_set, label_set) pairs."""
    if not pairs:
        return 0.0
    f1_values: list[float] = []
    for predicted, label in pairs:
        tp = len(predicted & label)
        pred_total = len(predicted)
        label_total = len(label)
        if pred_total == 0 and label_total == 0:
            f1_values.append(1.0)
            continue
        denominator = pred_total + label_total
        f1_values.append((2 * tp) / denominator if denominator > 0 else 0.0)
    return float(sum(f1_values) / len(f1_values))
