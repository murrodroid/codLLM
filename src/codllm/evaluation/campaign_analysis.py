"""Reproducible publication coverage and paired intervals from private frozen artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from codllm.config import Config as TrainingConfig
from codllm.evaluation.artifacts import load_manifest, write_json
from codllm.evaluation.bootstrap import Config as BootstrapConfig
from codllm.evaluation.bootstrap import align_predictions, bootstrap_scores, intervals
from codllm.evaluation.coverage import label_coverage, label_sets
from codllm.evaluation.provenance import content_digest
from codllm.evaluation.reference import cod_hash


@dataclass(frozen=True)
class Config:
    """Own offline campaign selection and output defaults."""

    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    campaigns: tuple[tuple[str, str, int], ...] = (
        ("screen", "cod", 10),
        ("screen-metadata", "metadata", 10),
        ("interaction-patience20", "cod", 20),
        ("interaction-metadata-patience20", "metadata", 20),
    )
    expected_cells: int = 8


def file_sha256(path: Path) -> str:
    """Hash a retained artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep only analyzed scalar metrics in the portable evidence ledger."""
    report = json.loads(json.dumps(report))
    keys = {f"pub_v1_{metric}" for metric in report["bootstrap"]["metrics"]}
    for run in report["runs"]:
        run["evidence"]["recorded_metrics"] = {
            key: value
            for key, value in run["evidence"]["recorded_metrics"].items()
            if key in keys
        }
    return report


def read_selected(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify selected prediction content against its original content-addressed identity."""
    payload = json.loads(path.read_text())
    prediction_path = path.parent / payload["predictions"]
    if prediction_path.parent != path.parent or prediction_path.suffix != ".parquet":
        raise ValueError(
            "Selected artifact must reference a sibling prediction parquet."
        )
    frame = pd.read_parquet(prediction_path)
    records = frame.to_dict("records")
    for row in records:
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
    if (
        content_digest({"identity": payload["identity"], "records": records})
        != payload["digest"]
    ):
        raise ValueError(f"Selected prediction content changed: {path}")
    return frame, {
        "selected_path": str(path),
        "selected_sha256": file_sha256(path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_path),
        "identity": payload["identity"],
        "recorded_metrics": payload["metrics"],
    }


def comparison_pairs(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Cover every within-campaign recipe pair plus matched input and patience changes."""
    comparisons = []
    for first, second in itertools.combinations(runs, 2):
        same_input = first["input"] == second["input"]
        same_patience = first["patience"] == second["patience"]
        same_recipe = first["cell"] == second["cell"]
        kind = None
        if same_input and same_patience:
            a, b = first["recipe"], second["recipe"]
            changed = [key for key in a if a[key] != b[key]]
            kind = changed[0] if len(changed) == 1 else "recipe_pair"
            if kind == "floor":
                first, second = sorted(
                    (first, second), key=lambda run: run["recipe"]["floor"]
                )
            elif kind in {"synthesis", "pretraining"}:
                first, second = sorted(
                    (first, second), key=lambda run: run["recipe"][kind], reverse=True
                )
        elif same_recipe and same_patience:
            kind = "metadata"
            first, second = sorted(
                (first, second),
                key=lambda run: run["input"] == "metadata",
                reverse=True,
            )
        elif same_recipe and same_input:
            kind = "patience"
            first, second = sorted(
                (first, second), key=lambda run: run["patience"], reverse=True
            )
        if kind:
            comparisons.append(
                {"first": first["id"], "second": second["id"], "kind": kind}
            )
    return comparisons


def write_tables(output: Path, report: dict[str, Any]) -> None:
    """Write aggregate comparison tables without private row-level contents."""
    records = [
        {
            "kind": comparison["kind"],
            "first": comparison["first"],
            "second": comparison["second"],
            "metric": metric,
            "difference_pp": result["estimate"] * 100,
            "ci95_low_pp": result["ci95"][0] * 100,
            "ci95_high_pp": result["ci95"][1] * 100,
        }
        for comparison in report["comparisons"]
        for metric, result in comparison["metrics"].items()
    ]
    pd.DataFrame(records).to_csv(output / "paired_intervals.csv", index=False)
    lines = [
        "# Paired publication intervals",
        "",
        (
            f"{report['bootstrap']['replicates']:,} paired connected-COD/source-record "
            f"{report['bootstrap']['weighting']} bootstrap replicates; "
            "95% percentile intervals. Differences are first minus second, in percentage points."
        ),
        "",
        (
            "Conditional on the selected models and observed archive panel. Validation selection and training-seed "
            "variation are not covered; intervals are exploratory and not multiplicity-adjusted."
        ),
        "",
        (
            "The published metrics are recomputed in every draw (pooled macro: reference ∪ predictions per model; "
            "source reference macro: reference labels per archive). Positive Bayesian weights retain observed support. "
            "Ordinary multinomial draws may omit rare labels; fixed-support sensitivity metrics assign absent labels "
            "zero F1. The CSV/JSON retain both definitions and rare-label diagnostics."
        ),
        "",
        report["method"],
        "",
        (
            "Bayesian draws assign normalized independent Exp(1) weights to connected groups, equivalent to "
            "Dirichlet(1,...,1) weights. Every model receives exactly the same group weights in each draw. "
            "We apply [Rubin's Bayesian bootstrap](https://people.eecs.berkeley.edu/~jordan/sail/readings/rubin.pdf) "
            "to the connected COD/source-record units, retaining row-frequency and equal-source metric definitions."
        ),
        "",
        (
            "The central interval of bootstrap metric draws need not contain the original point estimate: "
            "F1 is nonlinear and sparse support can shift its resampling distribution. Bayesian weighting does "
            "not supply missing evidence within labels represented by a single group. Treat these intervals as "
            "conditional exploratory evidence, not calibrated uncertainty over unseen label cases or new archives."
        ),
        "",
        "## Matched comparisons",
        "",
        (
            "| Change | First | Second | Pooled macro F1 Δ [95% interval] | Micro F1 Δ [95% interval] | "
            "Equal-source ref. macro F1 Δ [95% interval] |"
        ),
        "|---|---|---|---:|---:|---:|",
    ]
    for comparison in report["comparisons"]:
        if comparison["kind"] == "recipe_pair":
            continue
        cells = []
        for metric in ("macro_f1", "micro_f1", "source_mean_macro_f1_ref"):
            result = comparison["metrics"][metric]
            low, high = result["ci95"]
            cells.append(
                f"{100 * result['estimate']:+.3f} [{100 * low:+.3f}, {100 * high:+.3f}]"
            )
        lines.append(
            f"| {comparison['kind']} | {comparison['first']} | {comparison['second']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "All within-campaign recipe pairs, all metrics and sensitivity intervals are in "
                "[paired_intervals.csv](paired_intervals.csv)."
            ),
            "",
        ]
    )
    (output / "paired_intervals.md").write_text("\n".join(lines))


def analyze_publication(
    root: Path, output: Path, config: Config | None = None
) -> dict[str, Any]:
    """Audit frozen partitions and bootstrap all completed Phase 1 comparisons on a common panel."""
    config = config or Config()
    config.bootstrap.validate()
    output.mkdir(parents=True, exist_ok=True)
    frame_reference = None
    manifest_cache: dict[str, pd.DataFrame] = {}
    original_manifest_by_input: dict[str, str] = {}
    coverage, runs, draws = {}, [], {}
    for campaign, input_name, patience in config.campaigns:
        paths = sorted(
            (root / campaign).glob("*/run-*/publication/predictions/**/selected.json")
        )
        if len(paths) != config.expected_cells:
            raise ValueError(
                f"Expected {config.expected_cells} selected exports in {campaign}; found {len(paths)}."
            )
        seen_cells = set()
        for path in paths:
            directory = next(
                parent for parent in path.parents if parent.name == "publication"
            )
            run_dir = directory.parent
            cell = run_dir.parent.name
            if cell in seen_cells:
                raise ValueError(f"Duplicate selected export for {campaign}/{cell}.")
            seen_cells.add(cell)
            match = re.fullmatch(r"floor-(\d+)_synthetic-([\d.]+)_pretrain-(\d+)", cell)
            if match is None:
                raise ValueError(f"Unrecognized recipe identity: {cell}")
            run_id = f"{input_name}-p{patience}/{cell}"
            raw, evidence = read_selected(path)
            frame = align_predictions(raw, frame_reference)
            if frame_reference is None:
                frame_reference = frame
            contract = json.loads((directory / "contract.json").read_text())
            if contract["protocol"] != "cod":
                raise ValueError(
                    "Phase 1 analysis requires the frozen grouped-COD protocol."
                )
            if (
                contract["reference_fingerprint"]
                != evidence["identity"]["reference_fingerprint"]
            ):
                raise ValueError(
                    "Selected prediction reference differs from the frozen contract."
                )
            training_digest = contract["manifests"]["original_train"]["digest"]
            if input_name not in coverage:
                print(f"Verifying original manifests for {input_name}", flush=True)
                coverage[input_name] = label_coverage(directory)
                if len(coverage) > 1:
                    original = next(iter(coverage.values()))
                    other = coverage[input_name]
                    if original["training_identity_digest"] != other[
                        "training_identity_digest"
                    ] or any(
                        original["partitions"][split]["identity_digest"]
                        != other["partitions"][split]["identity_digest"]
                        for split in ("val", "test")
                    ):
                        raise ValueError(
                            "COD-only and metadata original partition identities differ."
                        )
                original_manifest_by_input[input_name] = training_digest
            if original_manifest_by_input[input_name] != training_digest:
                raise ValueError("Campaign original training partition differs.")
            digest = contract["manifests"]["val"]["digest"]
            if digest not in manifest_cache:
                manifest = (
                    load_manifest(directory, "val")
                    .sort_values("row_uid")
                    .reset_index(drop=True)
                )
                manifest_frame = manifest[["row_uid", "source_id", "record_id"]].copy()
                manifest_frame["cod_hash"] = manifest["cod_key"].map(cod_hash)
                manifest_frame["labels"] = label_sets(manifest, TrainingConfig())
                manifest_frame["predictions"] = manifest_frame["labels"]
                manifest_cache[digest] = align_predictions(manifest_frame)
            align_predictions(frame, manifest_cache[digest])
            print(f"Bootstrapping {run_id}", flush=True)
            cache_dir = output / "draws"
            cache_dir.mkdir(exist_ok=True)
            cache = cache_dir / f"{input_name}-p{patience}-{cell}.npz"
            cache_key = content_digest(
                {
                    "prediction_sha256": evidence["prediction_sha256"],
                    "bootstrap": asdict(config.bootstrap),
                    "implementation": file_sha256(
                        Path(__file__).with_name("bootstrap.py")
                    ),
                }
            )
            if cache.exists():
                with np.load(cache, allow_pickle=False) as stored:
                    if str(stored["key"]) != cache_key:
                        raise ValueError(
                            f"Stale bootstrap draws: use a fresh output directory, {cache}"
                        )
                    point, samples = stored["point"], stored["samples"]
                    diagnostics = json.loads(str(stored["diagnostics"]))
            else:
                point, samples, diagnostics = bootstrap_scores(frame, config.bootstrap)
                np.savez_compressed(
                    cache,
                    key=cache_key,
                    point=point,
                    samples=samples,
                    diagnostics=json.dumps(diagnostics),
                )
                cache.chmod(0o600)
            for i, metric in enumerate(config.bootstrap.metrics[:5]):
                recorded = evidence["recorded_metrics"][f"pub_v1_{metric}"]
                if not np.isclose(point[i], recorded, atol=1e-12, rtol=0):
                    raise ValueError(
                        f"Recovered {metric} differs from selected score for {run_id}."
                    )
            run = {
                "id": run_id,
                "input": input_name,
                "patience": patience,
                "cell": cell,
                "recipe": dict(
                    zip(
                        ("floor", "synthesis", "pretraining"),
                        map(float, match.groups()),
                        strict=True,
                    )
                ),
                "metrics": intervals(point, samples, config.bootstrap),
                "diagnostics": diagnostics,
                "evidence": evidence,
                "contract_sha256": file_sha256(directory / "contract.json"),
            }
            runs.append(run)
            draws[run_id] = point, samples
            write_json(output / "coverage.json", coverage)
            write_json(output / "partial_runs.json", runs)
    comparisons = comparison_pairs(runs)
    for comparison in comparisons:
        first, first_draws = draws[comparison["first"]]
        second, second_draws = draws[comparison["second"]]
        comparison["metrics"] = intervals(
            first - second, first_draws - second_draws, config.bootstrap
        )
    report = {
        "version": 1,
        "analyzed_at": datetime.now(ZoneInfo("Europe/Copenhagen")).isoformat(),
        "execution": {
            "host": socket.gethostname(),
            "lsf_job_id": os.getenv("LSB_JOBID"),
            "root": str(root),
        },
        "bootstrap": asdict(config.bootstrap),
        "method": (
            "Paired Bayesian connected COD/source-record cluster bootstrap, Dirichlet(1,...,1) group weights; "
            "95% equal-tail credible intervals on the observed empirical support."
            if config.bootstrap.weighting == "bayesian"
            else "Ordinary n-out-of-n paired connected COD/source-record cluster bootstrap; percentile confidence intervals."
        ),
        "scope": "Selected validation predictions; conditional on models and archive panel, exploratory, no seed variance.",
        "multiplicity": "Nominal per-comparison intervals; no family-wise coverage or post-selection correction.",
        "macro_support": "Published metric support; full-evaluation fixed-support sensitivity metrics included.",
        "runs": runs,
        "comparisons": comparisons,
    }
    write_json(output / "bootstrap.json", aggregate_report(report))
    write_tables(output, report)
    print(
        f"Completed {len(runs)} models and {len(comparisons)} paired comparisons in {output}",
        flush=True,
    )
    return report
