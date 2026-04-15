from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplits:
    """Container for train, validation, and test dataframe splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def resolve_training_frames(
    splits: DataSplits,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return train and optional validation dataframes from split output."""
    if splits.val.empty:
        return splits.train, None
    return splits.train, splits.val
