"""Regenerate publication figure examples from the reviewed aggregate ledgers."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from experiments.checkpoint_consistency import analyze


@dataclass(frozen=True)
class Config:
    """Own figure inputs and display settings; training configuration is unaffected."""

    output: Path = Path(__file__).resolve().parent
    inputs: tuple[str, ...] = (
        "checkpoint_consistency_input.json",
        "metadata_checkpoint_consistency_input.json",
    )
    names: tuple[str, ...] = ("COD-only", "COD+age+sex")
    colors: tuple[str, ...] = ("#0072B2", "#D55E00")
    markers: tuple[str, ...] = ("o", "s")
    metrics: tuple[str, ...] = (
        "pub_v1_source_mean_macro_f1_ref",
        "macro_f1",
        "micro_f1",
    )
    titles: tuple[str, ...] = (
        "Equal-source macro F1 (ref.)",
        "Pooled macro F1",
        "Micro F1",
    )
    dpi: int = 180
    bootstrap_input: str = "paired_bootstrap.json"
    bootstrap_sensitivity_input: str = "paired_bootstrap_multinomial.json"
    coverage_input: str = "label_coverage.json"


def load(
    config: Config,
) -> tuple[list[list[dict[str, Any]]], list[tuple[float, ...]], np.ndarray]:
    """Validate both ledgers and align scores by recipe, never by file order."""
    campaigns = []
    for filename in config.inputs:
        data = json.loads((config.output.parent / filename).read_text())
        analyze(data)
        campaigns.append({run["cell"]: run for run in data["runs"]})
    cells = sorted(campaigns[0])
    if len(cells) != 8 or set(cells) != set(campaigns[1]):
        raise ValueError("Expected the same eight recipes in both ledgers")
    runs = [[campaign[cell] for cell in cells] for campaign in campaigns]
    recipes = [
        tuple(
            map(
                float,
                re.fullmatch(
                    r"floor-(\d+)_synthetic-([\d.]+)_pretrain-(\d+)", cell
                ).groups(),
            )
        )
        for cell in cells
    ]
    scores = (
        np.array(
            [
                [
                    [
                        run["selected_metrics"][
                            key if key.startswith("pub_v1_") else f"pub_v1_{key}"
                        ]
                        for key in config.metrics
                    ]
                    for run in campaign
                ]
                for campaign in runs
            ]
        )
        * 100
    )
    if not np.isfinite(scores).all():
        raise ValueError("Selected scores must be finite")
    return runs, recipes, scores


def panels(rows: int, config: Config) -> tuple[plt.Figure, np.ndarray]:
    """Create three metric columns with consistent styling."""
    fig, axes = plt.subplots(
        rows, 3, figsize=(12, (2.6 if rows == 1 else 2) + 3.7 * rows), squeeze=False
    )
    fig.subplots_adjust(
        left=0.19,
        right=0.98,
        bottom=0.2 if rows == 1 else 0.12,
        top=0.82 if rows == 1 else 0.89,
        wspace=0.32,
        hspace=0.4,
    )
    for ax in axes.flat:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#E3E6E8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    for ax, title in zip(axes[0], config.titles, strict=True):
        ax.set_title(title, fontsize=11, pad=12)
    return fig, axes


def rows(ax: plt.Axes, labels: list[str]) -> None:
    """Keep recipe labels and ordering consistent across plots."""
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)


def save(
    fig: plt.Figure,
    name: str,
    title: str,
    note: str,
    config: Config,
    legend: bool = True,
) -> None:
    """Save compact captions with PNG previews and editable SVG copies."""
    fig.suptitle(title, x=0.02, ha="left", fontsize=15, y=0.98)
    fig.text(0.02, 0.02, note, fontsize=9)
    if legend:
        handles = [
            Line2D([], [], color=color, marker=marker, ls="", label=label)
            for color, marker, label in zip(
                config.colors, config.markers, config.names, strict=True
            )
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.97),
            ncol=2,
            frameon=False,
        )
    for extension in ("png", "svg"):
        path = config.output / f"{name}.{extension}"
        fig.savefig(
            path,
            dpi=config.dpi,
            facecolor="white",
            metadata={"Date": None} if extension == "svg" else None,
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def evidence(config: Config, *, sensitivity: bool = False) -> dict[str, Any]:
    """Read aggregate HPC bootstrap results without accessing private predictions."""
    filename = (
        config.bootstrap_sensitivity_input if sensitivity else config.bootstrap_input
    )
    report = json.loads((config.output.parent / filename).read_text())
    if (
        report["bootstrap"]["weighting"]
        != ("multinomial" if sensitivity else "bayesian")
        or len(report["runs"]) != 32
    ):
        raise ValueError(
            "Expected the complete paired Phase 1 analysis with the requested weighting."
        )
    return {
        "runs": {run["id"]: run for run in report["runs"]},
        "pairs": {
            (pair["first"], pair["second"]): pair["metrics"]
            for pair in report["comparisons"]
        },
        "replicates": report["bootstrap"]["replicates"],
    }


def run_id(campaign: int, patience: int, recipe: tuple[float, ...]) -> str:
    """Resolve a plotted recipe to its audited run identity."""
    floor, synthesis, pretrain = recipe
    return (
        f"{'cod' if campaign == 0 else 'metadata'}-p{patience}/"
        f"floor-{floor:g}_synthetic-{synthesis:g}_pretrain-{pretrain:g}"
    )


def metric_key(key: str) -> str:
    """Map existing figure metric names to the bootstrap ledger."""
    return key.removeprefix("pub_v1_")


def paired_interval(
    data: dict[str, Any], first: str, second: str, key: str
) -> tuple[float, float, float]:
    """Return an oriented paired estimate and interval in percentage points."""
    reverse = (first, second) not in data["pairs"]
    result = data["pairs"][(second, first) if reverse else (first, second)][
        metric_key(key)
    ]
    point = result["estimate"] * 100
    low, high = np.asarray(result["ci95"]) * 100
    return (-point, -high, -low) if reverse else (point, low, high)


def interval(
    ax: plt.Axes,
    values: tuple[float, float, float],
    y: float,
    campaign: int,
    config: Config,
) -> None:
    """Draw interval endpoints separately so a skewed interval need not contain its point estimate."""
    point, low, high = values
    color = config.colors[campaign]
    ax.hlines(y, low, high, color=color, lw=1.4)
    ax.vlines([low, high], y - 0.045, y + 0.045, color=color, lw=1)
    ax.scatter(point, y, color=color, marker=config.markers[campaign], s=30, zorder=3)


def ordinary_interval(
    ax: plt.Axes,
    values: tuple[float, float, float],
    y: float,
    campaign: int,
    config: Config,
) -> None:
    """Overlay ordinary multinomial confidence intervals as a visible method sensitivity."""
    _, low, high = values
    ax.hlines(
        y + 0.06,
        low,
        high,
        color=config.colors[campaign],
        lw=1,
        linestyles="dashed",
        alpha=0.8,
    )


def selected(
    scores: np.ndarray,
    labels: list[str],
    recipes: list[tuple[float, ...]],
    data: dict[str, Any],
    config: Config,
) -> None:
    """Connect matching input regimes at their selected checkpoints."""
    fig, axes = panels(1, config)
    for j, ax in enumerate(axes[0]):
        rows(ax, labels if j == 0 else [""] * len(labels))
        for i in range(len(labels)):
            ax.plot(scores[:, i, j], [i - 0.12, i + 0.12], color="#B8BEC2", lw=1)
        for c in range(2):
            for i, recipe in enumerate(recipes):
                result = data["runs"][run_id(c, 20, recipe)]["metrics"][
                    metric_key(config.metrics[j])
                ]
                interval(
                    ax,
                    (100 * result["estimate"], *(100 * np.asarray(result["ci95"]))),
                    i + (c - 0.5) * 0.24,
                    c,
                    config,
                )
        ax.margins(x=0.15)
        ax.set_xlabel("F1 (%)")
    axes[0, 0].set_ylabel("Floor / synthesis / pretraining epochs")
    save(
        fig,
        "selected",
        "Selected validation scores",
        "Patience 20 · selected validation · 95% Bayesian cluster-bootstrap score intervals\n"
        "Conditional on trained models; paired input differences are shown separately",
        config,
    )


def differences(
    recipes: list[tuple[float, ...]],
    data: dict[str, Any],
    sensitivity: dict[str, Any],
    config: Config,
    patience: int = 20,
) -> None:
    """Show conditional factor changes for both input regimes on shared difference axes."""
    fig, axes = panels(3, config)
    factors = [
        (0, 450, 0, "Floor: 450 → 0\nSynthesis / pretraining"),
        (1, 0.3, 0.6, "Synthesis: .30 → .60\nFloor / pretraining"),
        (2, 4, 48, "Pretraining: 4 → 48\nFloor / synthesis"),
    ]
    extent = np.zeros(3)
    for row, (factor, before, after, title) in enumerate(factors):
        pairs, labels = [], []
        for i, recipe in enumerate(recipes):
            if recipe[factor] == before:
                target = list(recipe)
                target[factor] = after
                pairs.append((i, recipes.index(tuple(target))))
                labels.append(
                    " / ".join(
                        f"{value:g}" for k, value in enumerate(recipe) if k != factor
                    )
                )
        for j, ax in enumerate(axes[row]):
            rows(ax, labels if j == 0 else [""] * len(labels))
            ax.axvline(0, color="#555555", lw=0.8)
            for c in range(2):
                for i, (a, b) in enumerate(pairs):
                    values = paired_interval(
                        data,
                        run_id(c, patience, recipes[b]),
                        run_id(c, patience, recipes[a]),
                        config.metrics[j],
                    )
                    extent[j] = max(extent[j], float(np.abs(values).max()))
                    interval(ax, values, i + (c - 0.5) * 0.22, c, config)
                    ordinary = paired_interval(
                        sensitivity,
                        run_id(c, patience, recipes[b]),
                        run_id(c, patience, recipes[a]),
                        config.metrics[j],
                    )
                    extent[j] = max(extent[j], float(np.abs(ordinary).max()))
                    ordinary_interval(ax, ordinary, i + (c - 0.5) * 0.22, c, config)
            ax.set_xlabel("Difference (percentage points)")
        axes[row, 0].set_ylabel(title)
    for j in range(3):
        for ax in axes[:, j]:
            ax.set_xlim(-extent[j] * 1.15, extent[j] * 1.15)
    save(
        fig,
        "differences" if patience == 20 else "differences_p10",
        f"Matched recipe differences · patience {patience}",
        "Paired 95% intervals: solid Bayesian; dashed ordinary bootstrap · positive favors the named change\n"
        "Exploratory validation comparisons; conditional on trained models, without multiplicity adjustment",
        config,
    )


def matched_inputs_and_patience(
    recipes: list[tuple[float, ...]],
    labels: list[str],
    data: dict[str, Any],
    sensitivity: dict[str, Any],
    config: Config,
) -> None:
    """Show paired age+sex and continuation differences with their own uncertainty."""
    fig, axes = panels(2, config)
    for row, patience in enumerate((10, 20)):
        for j, ax in enumerate(axes[row]):
            rows(ax, labels if j == 0 else [""] * len(labels))
            ax.axvline(0, color="#555555", lw=0.8)
            for i, recipe in enumerate(recipes):
                values = paired_interval(
                    data,
                    run_id(1, patience, recipe),
                    run_id(0, patience, recipe),
                    config.metrics[j],
                )
                interval(ax, values, i, 1, config)
                ordinary = paired_interval(
                    sensitivity,
                    run_id(1, patience, recipe),
                    run_id(0, patience, recipe),
                    config.metrics[j],
                )
                ordinary_interval(ax, ordinary, i, 1, config)
            ax.set_xlabel("Metadata − COD-only (percentage points)")
        axes[row, 0].set_ylabel(f"Patience {patience}\nFloor / synthesis / pretraining")
    save(
        fig,
        "metadata_differences",
        "Adding age and sex",
        "Paired 95% intervals: solid Bayesian; dashed ordinary bootstrap · selected validation · seed 777\n"
        "Age and sex tested together; intervals do not include training-seed variation",
        config,
        legend=False,
    )
    fig, axes = panels(1, config)
    for j, ax in enumerate(axes[0]):
        rows(ax, labels if j == 0 else [""] * len(labels))
        ax.axvline(0, color="#555555", lw=0.8)
        for c in range(2):
            for i, recipe in enumerate(recipes):
                values = paired_interval(
                    data,
                    run_id(c, 20, recipe),
                    run_id(c, 10, recipe),
                    config.metrics[j],
                )
                interval(ax, values, i + (c - 0.5) * 0.24, c, config)
                ordinary = paired_interval(
                    sensitivity,
                    run_id(c, 20, recipe),
                    run_id(c, 10, recipe),
                    config.metrics[j],
                )
                ordinary_interval(ax, ordinary, i + (c - 0.5) * 0.24, c, config)
        ax.set_xlabel("Patience 20 − 10 (percentage points)")
    axes[0, 0].set_ylabel("Floor / synthesis / pretraining")
    save(
        fig,
        "patience_differences",
        "Patience continuations",
        "Paired 95% intervals: solid Bayesian; dashed ordinary bootstrap · selected validation\n"
        "Inherited identical checkpoints have zero differences; continuations are not independent repeats",
        config,
    )


def coverage(config: Config) -> None:
    """Contrast distinct-label coverage with occurrence-weighted coverage of the frozen splits."""
    report = json.loads((config.output.parent / config.coverage_input).read_text())[
        "cod"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.27, top=0.77, wspace=0.52)
    for ax, split, title in zip(
        axes, ("val", "test"), ("Validation", "Test"), strict=True
    ):
        values = report["partitions"][split]["all"]
        numerator = [
            values["historically_seen_codes"],
            values["historically_seen_targets"],
        ]
        denominator = [values["distinct_codes"], values["targets"]]
        percentages = 100 * np.asarray(numerator) / denominator
        ax.barh([0, 1], percentages, color=config.colors[0], height=0.5)
        ax.barh(
            [0, 1],
            100 - percentages,
            left=percentages,
            color=config.colors[1],
            height=0.5,
        )
        for i, (num, den, percent) in enumerate(
            zip(numerator, denominator, percentages, strict=True)
        ):
            ax.text(
                1,
                i,
                f"{num:,} / {den:,}  ({percent:.3f}%)",
                va="center",
                color="white",
                fontsize=10,
            )
        ax.set_yticks([0, 1], ["Distinct labels", "Label occurrences"])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Present in original historical training (%)")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    save(
        fig,
        "label_coverage",
        "Unseen descriptions mostly contain known labels",
        "Checksum-verified original grouped-COD manifests · blue: seen in historical training; orange: absent\n"
        "All 35 historically unseen test labels occur in the recorded masterlist inventory; exposure is not accuracy",
        config,
        legend=False,
    )


def consistency(
    runs: list[list[dict[str, Any]]],
    scores: np.ndarray,
    labels: list[str],
    config: Config,
) -> None:
    """Plot actual checkpoint observations without presenting them as uncertainty intervals."""
    fig, axes = panels(2, config)
    for c, campaign in enumerate(runs):
        for j, key in enumerate(config.metrics):
            ax = axes[c, j]
            annotated = [
                label + (" *" if run["history_source"] == "inherited_parent" else "")
                for label, run in zip(labels, campaign, strict=True)
            ]
            rows(ax, annotated if j == 0 else [""] * len(labels))
            for i, run in enumerate(campaign):
                observations = {
                    point["epoch"]: point["metrics"][key] * 100 for point in run["rows"]
                }
                values = [
                    observations[epoch]
                    for epoch in range(run["best_epoch"] - 9, run["best_epoch"] + 1)
                ]
                ax.scatter(
                    values,
                    i + np.linspace(-0.12, 0.12, 10),
                    color="#8A9195",
                    s=12,
                    alpha=0.65,
                )
                ax.scatter(scores[c, i, j], i, color=config.colors[c], marker="D", s=30)
            ax.set_xlabel("F1 (%)")
        axes[c, 0].set_ylabel(config.names[c] + "\nFloor / synthesis / pretraining")
    for j in range(3):
        low = min(ax.get_xlim()[0] for ax in axes[:, j])
        high = max(ax.get_xlim()[1] for ax in axes[:, j])
        for ax in axes[:, j]:
            ax.set_xlim(low, high)
    save(
        fig,
        "consistency",
        "Checkpoint context",
        "Grey: B−9,…,B · diamond: selected score · * inherited parent window\n"
        "Patience 20 · correlated, selection-conditioned checkpoints; not seed replicates or confidence intervals",
        config,
        legend=False,
    )


def legacy(config: Config) -> None:
    """Render the earlier curves separately from the grouped-COD comparison."""
    text = (config.output.parent / "results.md").read_text()
    fig, axes = panels(1, config)
    fig.set_size_inches(12, 4.4)
    fig.subplots_adjust(left=0.08, bottom=0.24, top=0.76)
    specifications = [
        ("balance floor", "Balance floor"),
        ("synthesis", "Synthetic ratio"),
        ("pretraining dose", "Pretraining epochs"),
    ]
    for ax, (section, label) in zip(axes[0], specifications, strict=True):
        section_text = text.split(f"## Legacy: {section}\n", 1)[1].split("\n## ", 1)[0]
        values = []
        for line in section_text.splitlines():
            cells = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", line).strip("| ").split("|")
            if len(cells) == 6 and re.fullmatch(r"[\d.]+|Off", cells[0].strip()):
                values.append(
                    (
                        0 if cells[0].strip() == "Off" else float(cells[0]),
                        100 * float(cells[1]),
                    )
                )
        if len(values) != (6 if section == "pretraining dose" else 7):
            raise ValueError(f"Incomplete legacy table: {section}")
        ax.plot(*zip(*values), "o-", color=config.colors[0], lw=1, ms=4)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylim(70, 76)
    axes[0, 0].set_ylabel("Pooled macro F1 (%)")
    save(
        fig,
        "legacy",
        "Legacy response curves",
        "Row split · COD+age+sex · seed 777 · floor/dose scores provisional\n"
        "Intervals unavailable: original paired predictions and frozen manifests were not retained",
        config,
        legend=False,
    )


def main() -> None:
    """Create all examples using local aggregate evidence only."""
    config = Config()
    plt.rcParams.update(
        {"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "codllm-publication"}
    )
    runs, recipes, scores = load(config)
    data = evidence(config)
    sensitivity = evidence(config, sensitivity=True)
    labels = [
        f"{floor:g} / {synth:.2f} / {pretrain:g}" for floor, synth, pretrain in recipes
    ]
    selected(scores, labels, recipes, data, config)
    differences(recipes, data, sensitivity, config)
    differences(recipes, data, sensitivity, config, patience=10)
    matched_inputs_and_patience(recipes, labels, data, sensitivity, config)
    coverage(config)
    consistency(runs, scores, labels, config)
    legacy(config)
    print(f"Created eight PNG/SVG figure pairs in {config.output}")


if __name__ == "__main__":
    main()
