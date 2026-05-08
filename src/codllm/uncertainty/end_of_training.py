"""End-of-training pass: regenerate test, compute uncertainty, log RC curves to W&B.

Designed to run after `Trainer.evaluate(eval_dataset=test_ds)` so the
training-time eval metrics and the uncertainty pass agree on which model
checkpoint is being scored. Produces three downstream-ready artifacts:

  * predictions.jsonl - one row per test record with prediction + 5 signals
  * test_rows.parquet - the exact test split used (text / label / source_id)
  * rc_curves.json    - per-signal, per-metric coverage points

All three are logged as W&B artifacts so the cross-source generalization
analysis and selective-prediction notebooks become offline operations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from codllm.config import Config
from codllm.uncertainty.rc_curve import (
    DEFAULT_COVERAGES,
    UNCERTAINTY_SIGNAL_DIRECTION,
    UNCERTAINTY_SIGNAL_NAMES,
    RcCurvePoint,
    accuracy_metric,
    compute_rc_curve_by_coverage,
    sample_f1_metric,
)
from codllm.uncertainty.signals import (
    UncertaintySignals,
    compute_per_record_signals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (testable without a real model / W&B run)
# ---------------------------------------------------------------------------


def _split_codes(text: str, separator: str) -> set[str]:
    """Split a decoded label/prediction string into a normalized code set."""
    if not text:
        return set()
    if not separator:
        return {text.strip()} if text.strip() else set()
    return {token.strip() for token in text.split(separator) if token.strip()}


def _build_record_row(
    *,
    idx: int,
    source_text: str,
    gold: str,
    source_id: str,
    signals: UncertaintySignals,
    label_separator: str,
) -> dict[str, Any]:
    """Build one JSONL record from a generated prediction and gold label."""
    predicted_codes = _split_codes(signals.prediction, label_separator)
    gold_codes = _split_codes(gold, label_separator)
    correct = signals.prediction.strip() == gold.strip()
    exact_match = predicted_codes == gold_codes
    return {
        "idx": int(idx),
        "source_id": str(source_id),
        "source": str(source_text),
        "gold": str(gold),
        "prediction": signals.prediction,
        "correct": bool(correct),
        "exact_match": bool(exact_match),
        "n_tokens": int(signals.n_tokens),
        "sum_logprob": float(signals.sum_logprob),
        "mean_logprob": float(signals.mean_logprob),
        "min_logprob": float(signals.min_logprob),
        "mean_entropy": float(signals.mean_entropy),
        "first_token_entropy": float(signals.first_token_entropy),
    }


def _records_to_metric_inputs(
    records: list[dict[str, Any]],
    label_separator: str,
) -> dict[str, list[Any]]:
    """Pre-compute per-record inputs for the supported RC-curve metrics."""
    accuracies = [bool(r["correct"]) for r in records]
    exact_matches = [bool(r["exact_match"]) for r in records]
    sample_pairs = [
        (
            _split_codes(r["prediction"], label_separator),
            _split_codes(r["gold"], label_separator),
        )
        for r in records
    ]
    return {
        "accuracy": accuracies,
        "exact_match": exact_matches,
        "sample_f1": sample_pairs,
    }


_METRIC_FN_REGISTRY = {
    "accuracy": accuracy_metric,
    "exact_match": accuracy_metric,
    "sample_f1": sample_f1_metric,
}


def compute_all_rc_curves(
    records: list[dict[str, Any]],
    *,
    label_separator: str,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> dict[str, dict[str, list[RcCurvePoint]]]:
    """Build the full {metric_name: {signal_name: [RcCurvePoint, ...]}} table."""
    metric_inputs = _records_to_metric_inputs(records, label_separator)
    curves: dict[str, dict[str, list[RcCurvePoint]]] = {
        metric_name: {} for metric_name in metric_inputs
    }
    for signal_name in UNCERTAINTY_SIGNAL_NAMES:
        raw_scores = [float(r[signal_name]) for r in records]
        direction = UNCERTAINTY_SIGNAL_DIRECTION[signal_name]
        for metric_name, inputs in metric_inputs.items():
            metric_fn = _METRIC_FN_REGISTRY[metric_name]
            curves[metric_name][signal_name] = compute_rc_curve_by_coverage(
                raw_scores,
                inputs,
                metric_fn=metric_fn,
                direction=direction,
                coverages=coverages,
            )
    return curves


def _rc_curves_to_serializable(
    curves: Mapping[str, Mapping[str, list[RcCurvePoint]]],
) -> dict[str, dict[str, list[dict[str, float]]]]:
    """Convert the curve table to plain dicts for JSON serialization."""
    payload: dict[str, dict[str, list[dict[str, float]]]] = {}
    for metric_name, by_signal in curves.items():
        payload[metric_name] = {}
        for signal_name, points in by_signal.items():
            payload[metric_name][signal_name] = [asdict(point) for point in points]
    return payload


# ---------------------------------------------------------------------------
# I/O + W&B side effects
# ---------------------------------------------------------------------------


def write_predictions_jsonl(records: Iterable[dict[str, Any]], path: Path) -> Path:
    """Write per-record JSONL with predictions + uncertainty signals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_test_rows_parquet(test_df: pd.DataFrame, path: Path) -> Path:
    """Persist the exact test split rows used for end-of-training evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(path, index=False)
    return path


def write_rc_curves_json(
    curves: Mapping[str, Mapping[str, list[RcCurvePoint]]],
    path: Path,
) -> Path:
    """Persist the full RC-curve table for offline plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _rc_curves_to_serializable(curves)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _active_wandb() -> Any | None:
    """Return the active W&B module when a run is initialized, else None."""
    if os.getenv("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    if getattr(wandb, "run", None) is None:
        return None
    return wandb


def _log_rc_curves_to_wandb(
    curves: Mapping[str, Mapping[str, list[RcCurvePoint]]],
    *,
    namespace: str = "uncertainty",
) -> None:
    """Log RC curves as W&B tables and overlaid line plots, when W&B is active."""
    wandb = _active_wandb()
    if wandb is None:
        return

    payload: dict[str, Any] = {}
    flat_rows: list[list[Any]] = []
    for metric_name, by_signal in curves.items():
        coverage_axis: list[float] = []
        per_signal_values: dict[str, list[float]] = {}
        for signal_name, points in by_signal.items():
            per_signal_values[signal_name] = []
            for point in points:
                if not coverage_axis or len(coverage_axis) < len(points):
                    coverage_axis.append(float(point.coverage))
                per_signal_values[signal_name].append(float(point.metric_value))
                flat_rows.append(
                    [
                        metric_name,
                        signal_name,
                        float(point.coverage),
                        float(point.threshold),
                        int(point.n_kept),
                        float(point.metric_value),
                    ]
                )

        if coverage_axis and per_signal_values:
            ordered_signals = sorted(per_signal_values.keys())
            payload[f"{namespace}/rc_curve/{metric_name}"] = wandb.plot.line_series(
                xs=coverage_axis,
                ys=[per_signal_values[name] for name in ordered_signals],
                keys=list(ordered_signals),
                title=f"Risk-coverage: {metric_name}",
                xname="coverage",
            )

    flat_table = wandb.Table(
        columns=[
            "metric",
            "signal",
            "coverage",
            "threshold",
            "n_kept",
            "value",
        ],
        data=flat_rows,
    )
    payload[f"{namespace}/rc_curve_table"] = flat_table

    if payload:
        wandb.log(payload)


def _log_artifacts_to_wandb(
    *,
    artifact_name: str,
    file_paths: list[Path],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Bundle the produced files into one W&B artifact, when W&B is active."""
    wandb = _active_wandb()
    if wandb is None:
        return
    artifact = wandb.Artifact(
        name=artifact_name,
        type="evaluation_outputs",
        metadata=dict(metadata or {}),
    )
    for path in file_paths:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    wandb.log_artifact(artifact)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_end_of_training_uncertainty(
    cfg: Config,
    model: Any,
    tokenizer: Any,
    test_df: pd.DataFrame,
    *,
    run_dir: Path,
    device: str | None = None,
    log_every: int = 50,
) -> dict[str, Any] | None:
    """Score the test split with output_scores, save artifacts, log RC curves.

    Returns a summary dict with the artifact paths and curve payload, or None
    when the test split is empty. Skipped silently when ``model`` does not
    support generation (e.g. sequence-classification path).
    """
    if test_df is None or len(test_df) == 0:
        return None
    if not hasattr(model, "generate"):
        logger.info("Skipping uncertainty pass: model does not support .generate().")
        return None

    text_column = cfg.dataset_text_column
    label_column = cfg.dataset_label_column
    if text_column not in test_df.columns or label_column not in test_df.columns:
        logger.warning(
            "Skipping uncertainty pass: missing %r or %r columns in test split.",
            text_column,
            label_column,
        )
        return None

    has_source_id = "source_id" in test_df.columns
    resolved_device = device or _resolve_device_from_model(model)
    max_new_tokens = cfg.resolved_max_target_length()
    label_separator = cfg.label_separator

    was_training = getattr(model, "training", False)
    if was_training:
        model.eval()

    t_start = time.time()
    records: list[dict[str, Any]] = []
    total = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows()):
        source_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        gold = str(row[label_column]) if pd.notna(row[label_column]) else ""
        source_id = str(row["source_id"]) if has_source_id else ""
        signals = compute_per_record_signals(
            model,
            tokenizer,
            source_text,
            device=resolved_device,
            max_new_tokens=max_new_tokens,
        )
        records.append(
            _build_record_row(
                idx=idx,
                source_text=source_text,
                gold=gold,
                source_id=source_id,
                signals=signals,
                label_separator=label_separator,
            )
        )
        if log_every > 0 and (idx + 1) % log_every == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1e-6)
            logger.info(
                "uncertainty pass %d/%d  rate=%.1f rec/s",
                idx + 1,
                total,
                rate,
            )

    if was_training:
        model.train()

    run_dir = Path(run_dir)
    predictions_path = write_predictions_jsonl(
        records, run_dir / "predictions.jsonl"
    )
    test_rows_path = write_test_rows_parquet(
        test_df, run_dir / "test_rows.parquet"
    )

    curves = compute_all_rc_curves(records, label_separator=label_separator)
    rc_curves_path = write_rc_curves_json(curves, run_dir / "rc_curves.json")
    _log_rc_curves_to_wandb(curves)
    _log_artifacts_to_wandb(
        artifact_name="end_of_training_eval",
        file_paths=[predictions_path, test_rows_path, rc_curves_path],
        metadata={
            "n_records": len(records),
            "max_new_tokens": int(max_new_tokens),
            "label_separator": label_separator,
        },
    )

    return {
        "n_records": len(records),
        "predictions_path": str(predictions_path),
        "test_rows_path": str(test_rows_path),
        "rc_curves_path": str(rc_curves_path),
        "rc_curves": _rc_curves_to_serializable(curves),
    }


def _resolve_device_from_model(model: Any) -> str:
    """Return a device string suitable for `tokenizer(...).to(device)`."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"
    except Exception:
        return "cpu"
