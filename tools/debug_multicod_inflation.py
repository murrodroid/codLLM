"""Diagnose where the train-set multi-CoD inflation comes from.

Loads the processed parquet, replicates the train/val/test split the
production pipeline uses, and walks each transformation step to show the
label-count cardinality at each stage. Lets us pinpoint whether multi-CoD
rows are entering via synthesis or via floor upsampling on real compounds.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET = REPO_ROOT / "data" / "processed" / "data.parquet"

# Settings that match the size_sweep_*.toml configuration:
LABEL_SEPARATOR = ","
DATA_SEED = 333
TRAIN_RATIO = 0.98
VAL_RATIO = 0.01
TEST_RATIO = 0.01
FLOOR = 50
MAIN_SOURCES = {
    "belgium_1920_1930",
    "amsterdam_1854_1926",
    "copenhagen_may2025",
    "ipswich_1871_1911",
    "madrid_1905_1927",
}


def label_count(s: str) -> int:
    if not s:
        return 0
    return len([p for p in s.split(LABEL_SEPARATOR) if p.strip()])


def card_summary(df: pd.DataFrame, title: str) -> None:
    counts = df["label"].astype(str).map(label_count).value_counts().sort_index()
    total = int(counts.sum())
    print(f"\n{title} (rows={total:,})")
    for k, v in counts.items():
        print(f"  label_count={k}: {v:>10,} ({100*v/total:.1f}%)")


def analytic_post_upsample_label_counts(
    train_df: pd.DataFrame, floor: int
) -> dict[int, int]:
    """Return post-upsample label_count distribution analytically.

    Each label class with count<floor gets bumped to exactly `floor`. The
    label_count of those rows equals the label_count of the original class
    (since upsampled rows duplicate the source row's label). Bucketing by
    label_count tells us how upsample inflates the cardinality distribution.
    """
    labels = train_df["label"].astype(str)
    class_counts = labels.value_counts()
    per_label_card = {label: label_count(str(label)) for label in class_counts.index}

    # Start with the pre-upsample distribution.
    post = Counter()
    for label, c in class_counts.items():
        card = per_label_card[label]
        # After upsample: if c < floor, target = floor, else target = c.
        post[card] += max(int(c), floor)
    return dict(sorted(post.items()))


def main() -> int:
    if not PARQUET.exists():
        print(f"Parquet not found: {PARQUET}", file=sys.stderr)
        return 1

    df = pd.read_parquet(PARQUET, columns=["source_id", "label"])
    df = df[df["source_id"].isin(MAIN_SOURCES)].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows across {len(MAIN_SOURCES)} main sources")

    # Stage 1: pre-split distribution
    card_summary(df, "Stage 1: pre-split full data")

    # Stage 2: train/val/test split (matches DataHandler.split_dataframe with data_seed=333)
    holdout_ratio = VAL_RATIO + TEST_RATIO
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_ratio, random_state=DATA_SEED, shuffle=True
    )
    test_ratio = TEST_RATIO / holdout_ratio
    val_df, test_df = train_test_split(
        holdout_df, test_size=test_ratio, random_state=DATA_SEED, shuffle=True
    )
    card_summary(train_df, "Stage 2a: train AFTER split, BEFORE upsample")
    card_summary(val_df, "Stage 2b: val")
    card_summary(test_df, "Stage 2c: test")

    # Stage 3: analytic post-upsample distribution (no row materialization).
    post = analytic_post_upsample_label_counts(train_df, floor=FLOOR)
    total_post = sum(post.values())
    print(
        f"\nStage 3 (analytic): train AFTER floor={FLOOR} upsample "
        f"(rows={total_post:,})"
    )
    for k, v in post.items():
        print(f"  label_count={k}: {v:>10,} ({100*v/total_post:.1f}%)")

    # Bonus: count how many distinct compound labels exist and how many trigger upsample
    counts = train_df["label"].astype(str).value_counts()
    n_unique = len(counts)
    n_below_floor = int((counts < FLOOR).sum())
    n_single = int(
        sum(1 for label in counts.index if label_count(str(label)) == 1)
    )
    n_multi = int(
        sum(1 for label in counts.index if label_count(str(label)) >= 2)
    )
    n_single_below_floor = int(
        sum(
            1
            for label, c in counts.items()
            if c < FLOOR and label_count(str(label)) == 1
        )
    )
    n_multi_below_floor = int(
        sum(
            1
            for label, c in counts.items()
            if c < FLOOR and label_count(str(label)) >= 2
        )
    )
    print(f"\n--- Class statistics on TRAIN before upsample ---")
    print(f"  unique label classes: {n_unique:,}")
    print(f"    single-CoD classes:  {n_single:,}")
    print(f"    multi-CoD classes:   {n_multi:,}")
    print(f"  classes below floor={FLOOR}: {n_below_floor:,}")
    print(f"    single-CoD below floor: {n_single_below_floor:,}")
    print(f"    multi-CoD below floor:  {n_multi_below_floor:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
