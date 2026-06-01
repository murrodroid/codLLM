"""Top 30 ICD-10h codes restricted to the five main archival sources.

Excludes historic_strings_en_2024 (masterlist-derived reference text rather
than archival messy data) so the chart reflects what the model actually
trains/evaluates against for real historical death records.

Run from repo root:
    uv run python tools/plot_class_distribution_top30.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET = REPO_ROOT / "data" / "processed" / "data.parquet"
MASTERLIST = REPO_ROOT / "data" / "raw" / "ICD10h_Masterlist_2024.xlsx"
OUT_PATH = REPO_ROOT / "visualizations" / "class_distribution_top30_main_sources.png"

MAIN_SOURCES = {
    "belgium_1920_1930",
    "amsterdam_1854_1926",
    "copenhagen_may2025",
    "ipswich_1871_1911",
    "madrid_1905_1927",
}
TOP_N = 30
LABEL_SEPARATOR = ","
MAX_DESCRIPTION_CHARS = 50


def _load_descriptions() -> dict[str, str]:
    """Return ICD10h -> description map from the masterlist Excel."""
    df = pd.read_excel(MASTERLIST, sheet_name="Masterlist")
    desc = dict(zip(df["ICD10h"].astype(str), df["ICD10h_DESCRIPTION"].astype(str)))
    return desc


def main() -> int:
    if not PARQUET.exists():
        print(f"Parquet not found: {PARQUET}", file=sys.stderr)
        return 1

    df = pd.read_parquet(PARQUET, columns=["source_id", "label"])
    df = df[df["source_id"].isin(MAIN_SOURCES)]
    if df.empty:
        print("No rows match the main sources.", file=sys.stderr)
        return 1

    total_records = len(df)

    # Expand multi-CoD labels: split each row's label string into constituent codes.
    codes = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.split(LABEL_SEPARATOR)
        .explode()
        .str.strip()
    )
    codes = codes[codes != ""]

    code_counts = codes.value_counts()
    total_occurrences = int(code_counts.sum())
    top = code_counts.head(TOP_N)
    top_total = int(top.sum())
    top_codes = set(top.index)
    # Share of records that contain AT LEAST ONE of the top-N codes as a label.
    row_code_sets = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.split(LABEL_SEPARATOR)
        .map(lambda parts: {p.strip() for p in parts if p.strip()})
    )
    records_with_top_code = int(
        row_code_sets.map(lambda codes_set: bool(codes_set & top_codes)).sum()
    )
    coverage_pct = 100.0 * records_with_top_code / total_records

    descriptions = _load_descriptions()

    fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
    color = "#4C72B0"
    ax.barh(
        range(len(top)),
        top.values,
        color=color,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=10, family="monospace")
    ax.invert_yaxis()

    # Annotate each bar with its ICD-10h description in the blank space to the right.
    max_value = top.values.max()
    for i, (code, count) in enumerate(zip(top.index, top.values)):
        desc = descriptions.get(code, "")
        if len(desc) > MAX_DESCRIPTION_CHARS:
            desc = desc[: MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
        ax.text(
            count + max_value * 0.012,
            i,
            desc,
            va="center",
            fontsize=9,
            color="#333",
        )

    ax.set_xlabel("Records", fontsize=11)
    ax.set_xlim(0, max_value * 1.7)  # extra space for descriptions
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Secondary x-axis on top showing percentage of total records.
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda x: 100.0 * x / total_records,
            lambda p: p / 100.0 * total_records,
        ),
    )
    ax_top.set_xlabel("Share of records (%)", fontsize=11)
    ax_top.tick_params(axis="x", labelsize=9)

    total_records_m = total_records / 1_000_000
    fig.suptitle(
        f"Top {TOP_N} codes account for {coverage_pct:.0f}% of all "
        f"{total_records_m:.2f} million records",
        fontsize=14,
        fontweight="bold",
        x=0.05,
        y=0.985,
        ha="left",
    )

    plt.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.07)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(
        f"  total records (5 main sources): {total_records:,}\n"
        f"  total code occurrences: {total_occurrences:,}\n"
        f"  records containing >=1 top-{TOP_N} code: {records_with_top_code:,} "
        f"({coverage_pct:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
