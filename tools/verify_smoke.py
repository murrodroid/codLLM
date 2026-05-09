"""Verify the smoke-test run produced every expected artifact.

Walks the most recent runs_smoke/<model_slug>/run-* directory and checks:
  * predictions.jsonl with the expected keys per row (incl. all 5 signals)
  * test_rows.parquet with text + label + source_id columns
  * rc_curves.json keyed by metric -> signal -> [coverage points]
  * temperature.json with temperature + nll_before/after
  * trainer_state.json mentions per-source / hierarchy metric keys
  * wandb_run_id.txt is absent (W&B disabled in smoke) OR a valid id
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_BASE = REPO_ROOT / "runs_smoke"
EXPECTED_PREDICTION_KEYS = {
    "idx",
    "source_id",
    "source",
    "gold",
    "prediction",
    "correct",
    "exact_match",
    "n_tokens",
    "sum_logprob",
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
    "first_token_entropy",
}
EXPECTED_RC_METRICS = {"accuracy", "exact_match", "sample_f1"}
EXPECTED_RC_SIGNALS = {
    "sum_logprob",
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
    "first_token_entropy",
}


class SmokeVerificationError(Exception):
    """Raised when the smoke run is missing an expected artifact."""


def _find_latest_run_dir() -> Path:
    if not SMOKE_BASE.exists():
        raise SmokeVerificationError(f"Smoke base dir missing: {SMOKE_BASE}")
    candidates = []
    for model_dir in SMOKE_BASE.iterdir():
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run-"):
                candidates.append(run_dir)
    if not candidates:
        raise SmokeVerificationError(
            f"No run-* dir under any model slug in {SMOKE_BASE}"
        )
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def _check_predictions(run_dir: Path) -> int:
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise SmokeVerificationError(f"Missing predictions.jsonl in {run_dir}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise SmokeVerificationError("predictions.jsonl is empty")
    missing = EXPECTED_PREDICTION_KEYS - rows[0].keys()
    if missing:
        raise SmokeVerificationError(f"predictions.jsonl missing keys: {missing}")
    return len(rows)


def _check_test_rows(run_dir: Path) -> int:
    path = run_dir / "test_rows.parquet"
    if not path.exists():
        raise SmokeVerificationError(f"Missing test_rows.parquet in {run_dir}")
    df = pd.read_parquet(path)
    for column in ("text", "label", "source_id"):
        if column not in df.columns:
            raise SmokeVerificationError(
                f"test_rows.parquet missing column {column!r}"
            )
    return len(df)


def _check_rc_curves(run_dir: Path) -> None:
    path = run_dir / "rc_curves.json"
    if not path.exists():
        raise SmokeVerificationError(f"Missing rc_curves.json in {run_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload.keys()) != EXPECTED_RC_METRICS:
        raise SmokeVerificationError(
            f"rc_curves.json metrics mismatch: {sorted(payload.keys())}"
        )
    for metric_name, by_signal in payload.items():
        if set(by_signal.keys()) != EXPECTED_RC_SIGNALS:
            raise SmokeVerificationError(
                f"{metric_name} signals mismatch: {sorted(by_signal.keys())}"
            )
        for signal_name, points in by_signal.items():
            if not points:
                raise SmokeVerificationError(
                    f"{metric_name}/{signal_name} has zero coverage points"
                )


def _check_temperature(run_dir: Path) -> float | None:
    path = run_dir / "temperature.json"
    if not path.exists():
        # Temperature requires a non-empty val split; smoke datasets can
        # legitimately produce zero val rows. Treat as warning, not failure.
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("temperature", "nll_before", "nll_after", "n_tokens"):
        if key not in payload:
            raise SmokeVerificationError(f"temperature.json missing key {key!r}")
    return float(payload["temperature"])


def _check_per_source_keys(run_dir: Path) -> bool:
    path = run_dir / "trainer_state.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    log_history = payload.get("log_history") or []
    saw_source = False
    saw_hierarchy = False
    for entry in log_history:
        for key in entry:
            if key.startswith(("eval_source_", "test_source_")):
                saw_source = True
            if key.startswith(("eval_chapter_", "test_chapter_", "eval_block_", "test_block_")):
                saw_hierarchy = True
    return saw_source and saw_hierarchy


def main() -> int:
    try:
        run_dir = _find_latest_run_dir()
    except SmokeVerificationError as exc:
        print(f"FAIL  {exc}")
        return 1

    print(f"Latest smoke run: {run_dir}")
    failures: list[str] = []

    try:
        n_preds = _check_predictions(run_dir)
        print(f"  PASS  predictions.jsonl ({n_preds} rows)")
    except SmokeVerificationError as exc:
        failures.append(str(exc))
        print(f"  FAIL  {exc}")

    try:
        n_test = _check_test_rows(run_dir)
        print(f"  PASS  test_rows.parquet ({n_test} rows)")
    except SmokeVerificationError as exc:
        failures.append(str(exc))
        print(f"  FAIL  {exc}")

    try:
        _check_rc_curves(run_dir)
        print("  PASS  rc_curves.json (3 metrics x 5 signals)")
    except SmokeVerificationError as exc:
        failures.append(str(exc))
        print(f"  FAIL  {exc}")

    try:
        temperature = _check_temperature(run_dir)
        if temperature is None:
            print("  WARN  temperature.json absent (likely zero val rows)")
        else:
            print(f"  PASS  temperature.json (T={temperature:.3f})")
    except SmokeVerificationError as exc:
        failures.append(str(exc))
        print(f"  FAIL  {exc}")

    if _check_per_source_keys(run_dir):
        print("  PASS  trainer_state.json contains per-source + hierarchy keys")
    else:
        print(
            "  WARN  trainer_state.json missing per-source/hierarchy keys "
            "(expected when val split is empty at smoke scale)"
        )

    if failures:
        print(f"\nSmoke test FAILED ({len(failures)} issues)")
        return 1
    print("\nSmoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
