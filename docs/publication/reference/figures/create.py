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
    inputs: tuple[str, ...] = ("checkpoint_consistency_input.json", "metadata_checkpoint_consistency_input.json")
    names: tuple[str, ...] = ("COD-only", "COD+age+sex")
    colors: tuple[str, ...] = ("#0072B2", "#D55E00")
    markers: tuple[str, ...] = ("o", "s")
    metrics: tuple[str, ...] = ("pub_v1_source_mean_macro_f1_ref", "macro_f1", "micro_f1")
    titles: tuple[str, ...] = ("Equal-source macro F1 (ref.)", "Pooled macro F1", "Micro F1")
    dpi: int = 180


def load(config: Config) -> tuple[list[list[dict[str, Any]]], list[tuple[float, ...]], np.ndarray]:
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
    recipes = [tuple(map(float, re.fullmatch(r"floor-(\d+)_synthetic-([\d.]+)_pretrain-(\d+)", cell).groups()))
               for cell in cells]
    scores = np.array([[[run["selected_metrics"][key if key.startswith("pub_v1_") else f"pub_v1_{key}"]
                         for key in config.metrics] for run in campaign] for campaign in runs]) * 100
    if not np.isfinite(scores).all():
        raise ValueError("Selected scores must be finite")
    return runs, recipes, scores


def panels(rows: int, config: Config) -> tuple[plt.Figure, np.ndarray]:
    """Create three metric columns with consistent styling."""
    fig, axes = plt.subplots(rows, 3, figsize=(12, 2 + 3.7 * rows), squeeze=False)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.12, top=0.82 if rows == 1 else 0.89,
                        wspace=0.32, hspace=0.4)
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


def save(fig: plt.Figure, name: str, title: str, note: str, config: Config, legend: bool = True) -> None:
    """Save compact captions with PNG previews and editable SVG copies."""
    fig.suptitle(title, x=0.02, ha="left", fontsize=15, y=0.98)
    fig.text(0.02, 0.02, note, fontsize=9)
    if legend:
        handles = [Line2D([], [], color=color, marker=marker, ls="", label=label)
                   for color, marker, label in zip(config.colors, config.markers, config.names, strict=True)]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.97), ncol=2, frameon=False)
    for extension in ("png", "svg"):
        fig.savefig(config.output / f"{name}.{extension}", dpi=config.dpi, facecolor="white")
    plt.close(fig)


def selected(scores: np.ndarray, labels: list[str], config: Config) -> None:
    """Connect matching input regimes at their selected checkpoints."""
    fig, axes = panels(1, config)
    for j, ax in enumerate(axes[0]):
        rows(ax, labels if j == 0 else [""] * len(labels))
        for i in range(len(labels)):
            ax.plot(scores[:, i, j], [i, i], color="#B8BEC2", lw=1)
        for c in range(2):
            ax.scatter(scores[c, :, j], range(len(labels)), color=config.colors[c], marker=config.markers[c], s=35)
        ax.margins(x=0.15)
        ax.set_xlabel("F1 (%)")
    axes[0, 0].set_ylabel("Floor / synthesis / pretraining epochs")
    save(fig, "selected", "Selected validation scores",
         "Patience 20 · seed 777 · zoomed axes · no seed confidence intervals", config)


def differences(scores: np.ndarray, recipes: list[tuple[float, ...]], config: Config) -> None:
    """Show conditional factor changes for both input regimes on shared difference axes."""
    fig, axes = panels(3, config)
    factors = [(0, 450, 0, "Floor: 450 → 0\nSynthesis / pretraining"),
               (1, 0.3, 0.6, "Synthesis: .30 → .60\nFloor / pretraining"),
               (2, 4, 48, "Pretraining: 4 → 48\nFloor / synthesis")]
    extent = 0.0
    for row, (factor, before, after, title) in enumerate(factors):
        pairs, labels = [], []
        for i, recipe in enumerate(recipes):
            if recipe[factor] == before:
                target = list(recipe)
                target[factor] = after
                pairs.append((i, recipes.index(tuple(target))))
                labels.append(" / ".join(f"{value:g}" for k, value in enumerate(recipe) if k != factor))
        for j, ax in enumerate(axes[row]):
            rows(ax, labels if j == 0 else [""] * len(labels))
            ax.axvline(0, color="#555555", lw=0.8)
            for c in range(2):
                delta = np.array([scores[c, b, j] - scores[c, a, j] for a, b in pairs])
                extent = max(extent, float(np.abs(delta).max()))
                ax.scatter(delta, np.arange(len(labels)) + (c - 0.5) * 0.22,
                           color=config.colors[c], marker=config.markers[c], s=30)
            ax.set_xlabel("Difference (percentage points)")
        axes[row, 0].set_ylabel(title)
    for ax in axes.flat:
        ax.set_xlim(-extent * 1.15, extent * 1.15)
    save(fig, "differences", "Matched recipe differences",
         "Patience 20 · positive favors the named change · four recipe backgrounds are not seed replicates", config)


def consistency(runs: list[list[dict[str, Any]]], scores: np.ndarray, labels: list[str], config: Config) -> None:
    """Plot actual checkpoint observations without presenting them as uncertainty intervals."""
    fig, axes = panels(2, config)
    for c, campaign in enumerate(runs):
        for j, key in enumerate(config.metrics):
            ax = axes[c, j]
            annotated = [label + (" *" if run["history_source"] == "inherited_parent" else "")
                         for label, run in zip(labels, campaign, strict=True)]
            rows(ax, annotated if j == 0 else [""] * len(labels))
            for i, run in enumerate(campaign):
                observations = {point["epoch"]: point["metrics"][key] * 100 for point in run["rows"]}
                values = [observations[epoch] for epoch in range(run["best_epoch"] - 9, run["best_epoch"] + 1)]
                ax.scatter(values, i + np.linspace(-0.12, 0.12, 10), color="#8A9195", s=12, alpha=0.65)
                ax.scatter(scores[c, i, j], i, color=config.colors[c], marker="D", s=30)
            ax.set_xlabel("F1 (%)")
        axes[c, 0].set_ylabel(config.names[c] + "\nFloor / synthesis / pretraining")
    for j in range(3):
        low = min(ax.get_xlim()[0] for ax in axes[:, j])
        high = max(ax.get_xlim()[1] for ax in axes[:, j])
        for ax in axes[:, j]:
            ax.set_xlim(low, high)
    save(fig, "consistency", "Checkpoint context",
         "Grey: B−9,…,B · diamond: selected score · * inherited parent window\n"
         "Patience 20 · correlated, selection-conditioned checkpoints; not seed replicates or confidence intervals",
         config, legend=False)


def legacy(config: Config) -> None:
    """Render the earlier curves separately from the grouped-COD comparison."""
    text = (config.output.parent / "results.md").read_text()
    fig, axes = panels(1, config)
    fig.set_size_inches(12, 4.4)
    fig.subplots_adjust(left=0.08, bottom=0.24, top=0.76)
    specifications = [("balance floor", "Balance floor"), ("synthesis", "Synthetic ratio"),
                      ("pretraining dose", "Pretraining epochs")]
    for ax, (section, label) in zip(axes[0], specifications, strict=True):
        section_text = text.split(f"## Legacy: {section}\n", 1)[1].split("\n## ", 1)[0]
        values = []
        for line in section_text.splitlines():
            cells = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", line).strip("| ").split("|")
            if len(cells) == 6 and re.fullmatch(r"[\d.]+|Off", cells[0].strip()):
                values.append((0 if cells[0].strip() == "Off" else float(cells[0]), 100 * float(cells[1])))
        if len(values) != (6 if section == "pretraining dose" else 7):
            raise ValueError(f"Incomplete legacy table: {section}")
        ax.plot(*zip(*values), "o-", color=config.colors[0], lw=1, ms=4)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylim(70, 76)
    axes[0, 0].set_ylabel("Pooled macro F1 (%)")
    save(fig, "legacy", "Legacy response curves", "Row split · COD+age+sex · seed 777 · floor/dose scores provisional\n"
         "Lines connect observed settings; no fitted response or confidence band", config, legend=False)


def main() -> None:
    """Create all examples using local aggregate evidence only."""
    config = Config()
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "codllm-publication"})
    runs, recipes, scores = load(config)
    labels = [f"{floor:g} / {synth:.2f} / {pretrain:g}" for floor, synth, pretrain in recipes]
    selected(scores, labels, config)
    differences(scores, recipes, config)
    consistency(runs, scores, labels, config)
    legacy(config)
    print(f"Created four PNG/SVG figure pairs in {config.output}")


if __name__ == "__main__":
    main()
