"""Offline selected-checkpoint aggregation and paired group-bootstrap analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from codllm.evaluation.artifacts import write_json
from codllm.evaluation.bootstrap import (
    Config,
    align_predictions,
    bootstrap_scores,
    intervals,
)


def summarize_runs(root: str, output: str) -> pd.DataFrame:
    """Collect selected-checkpoint scalar summaries without mixing epoch maxima."""
    records = []
    for path in sorted(Path(root).rglob("selected.json")):
        payload = json.loads(path.read_text())
        identity = payload.get("identity", {})
        records.append({"summary_path": str(path), **identity, **payload["metrics"]})
    frame = pd.DataFrame(records)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    destination.chmod(0o600)
    return frame


def paired_bootstrap(
    first: str,
    second: str,
    output: str,
    replicates: int = Config.replicates,
    seed: int = Config.seed,
    weighting: str = Config.weighting,
) -> dict:
    """Bootstrap paired metric differences with fixed macro support and support sensitivity."""
    config = Config(replicates=replicates, seed=seed, weighting=weighting)
    a = align_predictions(pd.read_parquet(first))
    b = align_predictions(pd.read_parquet(second), a)
    first_point, first_samples, diagnostics = bootstrap_scores(a, config)
    second_point, second_samples, _ = bootstrap_scores(b, config)
    metrics = intervals(
        first_point - second_point, first_samples - second_samples, config
    )
    primary = metrics["source_mean_macro_f1_ref"]
    payload = {
        "difference_first_minus_second": primary["estimate"],
        "ci95": primary["ci95"],
        **diagnostics,
        "replicates": replicates,
        "seed": seed,
        "cluster_unit": "Connected COD and source-record groups",
        "estimand": "Conditional equal-source reference-supported macro F1; not training-seed or future-archive variance.",
        "weighting": weighting,
        "macro_support": "Published metric support; positive Bayesian weights preserve all observed labels.",
        "metrics": metrics,
    }
    write_json(Path(output), payload)
    return payload
