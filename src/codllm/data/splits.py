from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplits:
    """Container for train, validation, test, and optional held-out dataframe splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame | None = None
    holdout_eval: pd.DataFrame | None = None
    original_train: pd.DataFrame | None = None


def resolve_training_frames(
    splits: DataSplits,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return train and optional validation dataframes from split output."""
    if splits.val.empty:
        return splits.train, None
    return splits.train, splits.val
