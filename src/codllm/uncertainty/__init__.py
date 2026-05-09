"""Production-path uncertainty signals, RC curves, and end-of-training W&B logging."""

from codllm.uncertainty.rc_curve import (
    UNCERTAINTY_SIGNAL_NAMES,
    UNCERTAINTY_SIGNAL_DIRECTION,
    compute_rc_curve_by_coverage,
    coverage_at_threshold,
)
from codllm.uncertainty.signals import (
    UncertaintySignals,
    compute_per_record_signals,
)

__all__ = [
    "UNCERTAINTY_SIGNAL_DIRECTION",
    "UNCERTAINTY_SIGNAL_NAMES",
    "UncertaintySignals",
    "compute_per_record_signals",
    "compute_rc_curve_by_coverage",
    "coverage_at_threshold",
]
