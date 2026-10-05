"""Reproduce aggregate best-ending ten-epoch consistency statistics without network access."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median, pvariance
from typing import Any


def summarize(values: list[float], *, lower_is_better: bool = False) -> dict[str, float]:
    """Describe ten checkpoints using the publication B-9 through B specification."""
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise ValueError("Exactly ten finite observations are required")
    mean = fmean(values)
    variance = pvariance(values)
    center = (len(values) - 1) / 2
    slope = sum((i - center) * (value - mean) for i, value in enumerate(values)) / sum(
        (i - center) ** 2 for i in range(len(values))
    )
    uplift = values[-1] - median(values[:-1])
    return {
        "endpoint": values[-1],
        "mean": mean,
        "median": median(values),
        "variance": variance,
        "sd": math.sqrt(variance),
        "minimum": min(values),
        "maximum": max(values),
        "endpoint_uplift": -uplift if lower_is_better else uplift,
        "slope_per_epoch": slope,
    }


def analyze_run(run: dict[str, Any], metric_keys: list[str]) -> dict[str, Any]:
    """Validate checkpoint lineage and summarize each fully observed metric window."""
    best = run["best_epoch"]
    if not run["complete"] or best < 10 or int(best) != best:
        raise ValueError("A completed run and integral best epoch >= 10 are required")
    inherited = run["history_source"] == "inherited_parent"
    if inherited and best != run["anchor_epoch"]:
        raise ValueError("An inherited window must end at the parent anchor")
    if not inherited and (run["history_source"] != "child" or best - 9 <= run["anchor_epoch"]):
        raise ValueError("Child window crosses the fork or has an unknown history source")
    by_epoch = {}
    for row in run["rows"]:
        epoch = row["epoch"]
        if int(epoch) != epoch or not best - 9 <= epoch <= best:
            raise ValueError("Partial or out-of-window epoch")
        if epoch in by_epoch and row != by_epoch[epoch]:
            raise ValueError("Conflicting duplicate evaluation")
        by_epoch[epoch] = row
    expected = list(range(best - 9, best + 1))
    if sorted(by_epoch) != expected:
        raise ValueError("Missing consecutive full-epoch evaluations")
    rows = [by_epoch[epoch] for epoch in expected]
    steps_per_epoch = run["max_steps"] / run["epochs_limit"]
    if any(row["step"] != row["epoch"] * steps_per_epoch for row in rows):
        raise ValueError("Epoch/optimizer-step boundary mismatch")
    if rows[-1]["step"] != run["best_step"]:
        raise ValueError("Window endpoint does not match selected checkpoint step")
    if not math.isclose(rows[-1]["metrics"]["macro_f1"], run["best_metric"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("Endpoint macro F1 does not match selected checkpoint metric")
    summaries = {}
    for key in metric_keys:
        values = [row["metrics"].get(key) for row in rows]
        if any(value is None or not math.isfinite(value) for value in values):
            summaries[key] = {"status": "unavailable", "reason": "Missing or nonfinite epoch metric"}
            continue
        summary = summarize(values, lower_is_better=key == "loss")
        selected = run["selected_metrics"].get(key, run["selected_metrics"].get(f"pub_v1_{key}"))
        delta = None if selected is None else selected - values[-1]
        summaries[key] = {
            "status": "complete" if delta is None or abs(delta) <= 1e-12 else "selected_discrepancy",
            **summary,
            "restored_selected": selected,
            "restored_minus_endpoint": delta,
        }
    return {**run, "rows": rows, "window": [best - 9, best], "summaries": summaries}


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    """Check shared validation membership and retain provenance with all summaries."""
    identities = {json.dumps(run["contract"]["manifests"]["val"], sort_keys=True) for run in data["runs"]}
    if len(identities) != 1:
        raise ValueError("Runs do not share the same frozen validation population")
    return {**data, "runs": [analyze_run(run, data["metric_keys"]) for run in data["runs"]]}


def render(report: dict[str, Any]) -> str:
    """Render every reported metric with explicit units and fixed recipe identities."""
    lines = [
        "# Checkpoint consistency: computed tables",
        "",
        "Window: B-9 through B. Scores are percentages; SD/uplift are percentage points; slope is pp/epoch.",
        "Loss uses native units. Variance and full-precision observations are retained in the JSON report.",
        "Endpoint uplift compares B with the preceding nine-epoch median. SD is descriptive, not a CI.",
        "Selected is the original epoch-B score; available restored-best scores are checked separately in JSON.",
        "",
        "| ID | Floor | Synthesis | Pretraining | Window | History |",
        "|---|---:|---:|---:|---|---|",
    ]
    for index, run in enumerate(report["runs"], 1):
        parts = run["cell"].split("_")
        floor, synthesis, pretraining = [part.split("-")[-1] for part in parts]
        start, end = run["window"]
        url = f"https://wandb.ai/codllmdev/codllm/runs/{run['child_wandb_id']}"
        lines.append(
            f"| [R{index}]({url}) | {floor} | {synthesis} | {pretraining} | {start}–{end} | "
            f"{run['history_source']} |"
        )
    for key in report["metric_keys"]:
        lines.extend([
            "", f"## {key}", "",
            "| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for index, run in enumerate(report["runs"], 1):
            summary = run["summaries"][key]
            if summary["status"] == "unavailable":
                lines.append(f"| R{index} | unavailable | — | — | — | — | — | — | — |")
                continue
            scale = 1 if key == "loss" else 100
            fields = ["endpoint", "mean", "median", "sd", "minimum", "maximum", "endpoint_uplift", "slope_per_epoch"]
            formatted = " | ".join(f"{summary[field] * scale:.4f}" for field in fields)
            lines.append(f"| R{index} | {formatted} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Recompute JSON and Markdown reports from a previously recovered aggregate ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.input.resolve(), args.output_json.resolve(), args.output_md.resolve()]
    if len(set(paths)) != 3:
        parser.error("Input and output paths must be distinct")
    raw = args.input.read_bytes()
    report = analyze(json.loads(raw))
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    for path in (args.output_json, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output_md.write_text(render(report))
    print(f"Reported {len(report['runs'])} windows; input SHA256 {report['input_sha256']}")


if __name__ == "__main__":
    main()
