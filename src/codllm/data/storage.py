from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from codllm.config import Config
from codllm.input import DatasetMapping, build_processed_dataset

PROCESSING_METADATA_VERSION = 9
DEFAULT_PROCESSED_LOCK_TIMEOUT_SECONDS = 900.0
NON_PROCESSING_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "balance_strategy",
        "balance_target_quantile",
        "balance_perturbations",
        "balance_perturbations_per_sample",
        "balance_perturbation_mean",
        "balance_perturbation_variance",
        "balance_upsample_labels",
        "balance_upsample_perturbation_rate",
        "balance_upsample_inverse_power",
        "balance_upsample_budget_ratio",
        "balance_base_perturbation_rate",
        "masterlist_inject_enabled",
        "masterlist_inject_target_per_label",
        "masterlist_inject_perturbations",
        "masterlist_inject_perturbations_per_sample",
    }
)


def _normalize_processing_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Drop metadata keys that do not affect processed dataset content."""
    normalized = dict(metadata)
    for key in NON_PROCESSING_METADATA_KEYS:
        normalized.pop(key, None)
    return normalized


def save_processed_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Persist processed data as CSV or Parquet based on file extension."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    temporary_output = output.parent / f".{output.name}.{uuid4().hex}.tmp{suffix}"
    try:
        if suffix == ".csv":
            df.to_csv(temporary_output, index=False)
        elif suffix == ".parquet":
            df.to_parquet(temporary_output, index=False)
        else:
            raise ValueError("Unsupported processed file format. Use .csv or .parquet.")
        temporary_output.replace(output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def build_and_save_processed_dataset(
    cfg: Config,
    mapping_registry: Mapping[str, DatasetMapping] | None = None,
) -> pd.DataFrame:
    """Build and persist processed data using config output settings."""
    processed_df = build_processed_dataset(cfg, mapping_registry=mapping_registry)
    output_path = str(Path(cfg.data_processed_dir) / cfg.processed_filename)
    save_processed_dataset(processed_df, output_path)
    return processed_df
