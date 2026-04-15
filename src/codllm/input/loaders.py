from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from codllm.runtime.paths import resolve_source_path
from codllm.settings.options import SUPPORTED_TRAINING_INPUTS
from codllm.settings.schema import Config, DataSourceConfig
from codllm.input.harmonization import _harmonize_processed_labels
from codllm.input.mappings import DatasetMapping, MAPPING_REGISTRY
from codllm.input.transform import (
    _build_label,
    _build_text,
    _build_y,
    _get,
    _normalize_training_input,
    _processed_columns,
)


def _detect_file_type(path: Path, source: DataSourceConfig) -> str:
    """Determine file type from source config or file extension."""
    if source.file_type is not None:
        return source.file_type.lower().lstrip(".")
    return path.suffix.lower().lstrip(".")


def _read_raw_dataframe(path: Path, source: DataSourceConfig) -> pd.DataFrame:
    """Read one raw source file into a dataframe."""
    file_type = _detect_file_type(path, source)
    if file_type in {"csv", "txt", "tsv"}:
        return pd.read_csv(
            path,
            header=source.header,
            dtype=str,
            sep=source.sep,
            encoding=source.encoding,
            on_bad_lines="skip",
        )
    if file_type in {"xlsx", "xls"}:
        return pd.read_excel(
            path,
            header=source.header,
            dtype=str,
            sheet_name=source.sheet_name,
        )
    raise ValueError(
        f"Unsupported file type '{file_type}' for source '{source.source_id}'."
    )


def _select_rows_with_valid_labels(
    raw_df: pd.DataFrame,
    mapping: DatasetMapping,
    max_labels: int,
    drop_missing_label: bool,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Filter raw rows before dataset assembly based on resolved label availability."""
    y_codes = raw_df.apply(lambda row: _build_y(row, mapping), axis=1)
    label_counts = y_codes.apply(len)
    keep_mask = label_counts <= max_labels
    if drop_missing_label:
        keep_mask = keep_mask & (label_counts > 0)

    filtered_raw_df = raw_df.loc[keep_mask].copy()
    source_row_indices = filtered_raw_df.index.tolist()
    filtered_raw_df = filtered_raw_df.reset_index(drop=True)
    filtered_y_codes = y_codes.loc[keep_mask].reset_index(drop=True)
    return filtered_raw_df, filtered_y_codes, source_row_indices


def load_source_dataset(
    source: DataSourceConfig,
    mapping: DatasetMapping,
    training_input: Sequence[str],
    max_labels: int = 1,
    label_separator: str | None = None,
    text_field_separator: str | None = None,
    data_raw_dir: str | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    drop_missing_label: bool = True,
) -> pd.DataFrame:
    """Load and process one source dataset into the canonical schema."""
    if max_labels < 1:
        raise ValueError("max_labels must be at least 1.")
    default_cfg = Config()
    normalized_training_input = _normalize_training_input(training_input)
    effective_label_separator = (
        default_cfg.label_separator if label_separator is None else label_separator
    )
    effective_text_field_separator = (
        default_cfg.text_field_separator
        if text_field_separator is None
        else text_field_separator
    )
    effective_data_raw_dir = (
        default_cfg.data_raw_dir if data_raw_dir is None else data_raw_dir
    )
    effective_text_column = (
        default_cfg.dataset_text_column if text_column is None else text_column
    )
    effective_label_column = (
        default_cfg.dataset_label_column if label_column is None else label_column
    )
    source_path = resolve_source_path(source.path, effective_data_raw_dir)
    raw_df = _read_raw_dataframe(source_path, source)
    combined_skip_rows = sorted(set(mapping.skip_rows + source.skip_rows))
    if combined_skip_rows:
        raw_df = raw_df.drop(index=combined_skip_rows, errors="ignore").reset_index(
            drop=True
        )
    filtered_raw_df, filtered_y_codes, source_row_indices = (
        _select_rows_with_valid_labels(
            raw_df=raw_df,
            mapping=mapping,
            max_labels=max_labels,
            drop_missing_label=drop_missing_label,
        )
    )

    result = pd.DataFrame()
    result["source_id"] = [source.source_id] * len(filtered_raw_df)
    if mapping.record_id_col is None:
        result["record_id"] = [
            f"{source.source_id}:{source_idx}" for source_idx in source_row_indices
        ]
    else:
        extracted_ids = filtered_raw_df.apply(
            lambda row: _get(row, mapping.record_id_col),
            axis=1,
        )
        result["record_id"] = [
            record_id if record_id is not None else f"{source.source_id}:{source_idx}"
            for source_idx, record_id in zip(
                source_row_indices,
                extracted_ids.tolist(),
                strict=True,
            )
        ]
    result["source_path"] = [str(source_path)] * len(filtered_raw_df)
    result[effective_text_column] = filtered_raw_df.apply(
        lambda row: _build_text(
            row,
            mapping,
            normalized_training_input,
            field_separator=effective_text_field_separator,
        ),
        axis=1,
    )
    result["y_codes"] = filtered_y_codes
    result[effective_label_column] = result["y_codes"].apply(
        lambda codes: _build_label(codes, separator=effective_label_separator)
    )
    return result


def build_processed_dataset(
    cfg: Config,
    mapping_registry: Mapping[str, DatasetMapping] | None = None,
) -> pd.DataFrame:
    """Build one processed dataframe from all configured raw data sources."""
    registry = dict(MAPPING_REGISTRY)
    if mapping_registry is not None:
        registry.update(mapping_registry)

    processed_frames: list[pd.DataFrame] = []
    for source in cfg.data_sources:
        if not source.enabled:
            continue
        if source.mapping_id not in registry:
            raise KeyError(
                f"Unknown mapping_id '{source.mapping_id}' for source '{source.source_id}'."
            )
        mapping = registry[source.mapping_id]
        processed_frames.append(
            load_source_dataset(
                source=source,
                mapping=mapping,
                training_input=cfg.training_input,
                max_labels=cfg.max_label_count,
                label_separator=cfg.label_separator,
                text_field_separator=cfg.text_field_separator,
                data_raw_dir=cfg.data_raw_dir,
                text_column=cfg.dataset_text_column,
                label_column=cfg.dataset_label_column,
            )
        )

    if not processed_frames:
        return pd.DataFrame(
            columns=_processed_columns(
                cfg.dataset_text_column,
                cfg.dataset_label_column,
            )
        )
    processed = pd.concat(processed_frames, ignore_index=True)
    if cfg.label_harmonization_enabled:
        processed = _harmonize_processed_labels(cfg=cfg, dataframe=processed)
    return processed


def load_dataset(
    path: str,
    mapping: DatasetMapping,
    training_input: Sequence[str] | None = None,
    max_labels: int = 1,
    sep: str = ",",
) -> pd.DataFrame:
    """Load one dataset into legacy text/y output format."""
    default_cfg = Config()
    source = DataSourceConfig(
        source_id="inline_source",
        path=path,
        mapping_id="inline_mapping",
        sep=sep,
    )
    processed = load_source_dataset(
        source=source,
        mapping=mapping,
        training_input=training_input or list(SUPPORTED_TRAINING_INPUTS),
        max_labels=max_labels,
        data_raw_dir="",
        text_column=default_cfg.dataset_text_column,
        label_column=default_cfg.dataset_label_column,
        drop_missing_label=False,
    )
    return pd.DataFrame(
        {
            default_cfg.dataset_text_column: processed[default_cfg.dataset_text_column],
            "y": processed["y_codes"],
        }
    )
