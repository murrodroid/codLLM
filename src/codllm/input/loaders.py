from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from codllm.runtime.paths import resolve_source_path
from codllm.settings.options import SUPPORTED_TRAINING_INPUTS
from codllm.settings.schema import Config, DataSourceConfig
from codllm.settings.types import TrainingInput
from codllm.input.harmonization import _harmonize_processed_labels
from codllm.input.mappings import DatasetMapping, MAPPING_REGISTRY
from codllm.input.transform import (
    MISSING_VALUE_MARKERS,
    UNKNOWN_VALUE,
    _normalize_training_input,
    _omit_cod_prefix_for_input,
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


def _empty_raw_series(raw_df: pd.DataFrame) -> pd.Series:
    """Return a null object series aligned to a raw dataframe."""
    return pd.Series([None] * len(raw_df), index=raw_df.index, dtype=object)


def _raw_column(raw_df: pd.DataFrame, col: int | None) -> pd.Series:
    """Return a raw dataframe column by position, or nulls when unavailable."""
    if col is None or col < 0 or col >= len(raw_df.columns):
        return _empty_raw_series(raw_df)
    return raw_df.iloc[:, col]


def _normalize_raw_values(values: pd.Series) -> pd.Series:
    """Normalize source values using the same missing-value rules as row access."""
    missing_mask = values.isna()
    normalized = values.astype("string").str.strip()
    invalid_mask = (
        missing_mask
        | normalized.isna()
        | normalized.eq("")
        | normalized.str.lower().isin(MISSING_VALUE_MARKERS)
    )
    return normalized.astype(object).mask(invalid_mask, None)


def _format_age_values(values: pd.Series) -> pd.Series:
    """Normalize raw age values into compact numeric strings."""
    normalized = _normalize_raw_values(values)
    numeric = pd.to_numeric(normalized, errors="coerce").round(2)
    formatted: list[str] = []
    for value in numeric.tolist():
        if pd.isna(value):
            formatted.append(UNKNOWN_VALUE)
        elif float(value).is_integer():
            formatted.append(str(int(value)))
        else:
            formatted.append(f"{float(value):.2f}".rstrip("0").rstrip("."))
    return pd.Series(formatted, index=values.index, dtype=object)


def _format_sex_values(values: pd.Series, sex_map: Mapping[str, str]) -> pd.Series:
    """Map raw sex values into canonical values."""
    normalized = _normalize_raw_values(values)
    result = normalized.map(sex_map)
    missing_mask = result.isna()
    if missing_mask.any():
        lowered = normalized[missing_mask].astype("string").str.lower()
        result.loc[missing_mask] = lowered.map(sex_map)
    return result.fillna(UNKNOWN_VALUE).astype(object)


def _normalize_code_columns(raw_df: pd.DataFrame, columns: Sequence[int]) -> pd.DataFrame:
    """Return normalized code columns selected by source positions."""
    selected = {
        col: _normalize_raw_values(_raw_column(raw_df, col))
        for col in columns
        if col >= 0 and col < len(raw_df.columns)
    }
    if not selected:
        return pd.DataFrame(index=raw_df.index)
    return pd.DataFrame(selected, index=raw_df.index)


def _unique_codes_from_row(values: tuple[object, ...]) -> list[str]:
    """Collect unique non-empty code strings from tuple values in order."""
    codes: list[str] = []
    seen_codes: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        code = str(value)
        if code not in seen_codes:
            codes.append(code)
            seen_codes.add(code)
    return codes


def _build_y_codes(raw_df: pd.DataFrame, mapping: DatasetMapping, max_labels: int) -> pd.Series:
    """Build target code lists without constructing pandas row Series objects."""
    single_codes = _normalize_raw_values(_raw_column(raw_df, mapping.single_code_col))
    if max_labels == 1:
        return single_codes.map(lambda code: [] if code is None else [str(code)])

    multi_codes = _normalize_code_columns(raw_df, mapping.multi_code_cols or [])
    y_codes: list[list[str]] = []
    if multi_codes.empty:
        for code in single_codes.tolist():
            y_codes.append([] if code is None else [str(code)])
        return pd.Series(y_codes, index=raw_df.index, dtype=object)

    for row_codes, single_code in zip(
        multi_codes.itertuples(index=False, name=None),
        single_codes.tolist(),
        strict=True,
    ):
        codes = _unique_codes_from_row(row_codes)
        if not codes and single_code is not None:
            codes = [str(single_code)]
        y_codes.append(codes)
    return pd.Series(y_codes, index=raw_df.index, dtype=object)


def _build_text_values(
    raw_df: pd.DataFrame,
    mapping: DatasetMapping,
    training_input: Sequence[TrainingInput],
    field_separator: str,
    input_field_prefixes: Mapping[TrainingInput, str],
) -> pd.Series:
    """Build model input text with column-wise source extraction."""
    parts: list[pd.Series] = []
    omit_cod_prefix = _omit_cod_prefix_for_input(training_input)
    for feature in training_input:
        prefix = input_field_prefixes[feature]
        if feature == "cod":
            cod_values = _normalize_raw_values(_raw_column(raw_df, mapping.text_col))
            cod_values = cod_values.fillna(UNKNOWN_VALUE)
            cod_values = cod_values.str.replace(r"(?<=\w)\.(?=\w)", " ", regex=True)
            parts.append(cod_values if omit_cod_prefix else prefix + cod_values)
        elif feature == "age":
            age_values = _format_age_values(_raw_column(raw_df, mapping.age_col))
            parts.append(prefix + age_values)
        else:
            sex_values = _format_sex_values(
                _raw_column(raw_df, mapping.sex_col),
                mapping.sex_map,
            )
            parts.append(prefix + sex_values)

    if not parts:
        return pd.Series([""] * len(raw_df), index=raw_df.index, dtype=object)
    text = parts[0]
    for part in parts[1:]:
        text = text.str.cat(part, sep=field_separator)
    return text.astype(object)


def _select_rows_with_valid_labels(
    raw_df: pd.DataFrame,
    mapping: DatasetMapping,
    max_labels: int,
    drop_missing_label: bool,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Filter raw rows before dataset assembly based on resolved label availability."""
    y_codes = _build_y_codes(raw_df, mapping=mapping, max_labels=max_labels)
    label_counts = y_codes.str.len()
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
    input_field_prefixes: Mapping[TrainingInput, str] | None = None,
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
    effective_input_field_prefixes = (
        default_cfg.input_field_prefixes
        if input_field_prefixes is None
        else input_field_prefixes
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
        extracted_ids = _normalize_raw_values(
            _raw_column(filtered_raw_df, mapping.record_id_col)
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
    result[effective_text_column] = _build_text_values(
        raw_df=filtered_raw_df,
        mapping=mapping,
        training_input=normalized_training_input,
        field_separator=effective_text_field_separator,
        input_field_prefixes=effective_input_field_prefixes,
    )
    result["y_codes"] = filtered_y_codes
    result[effective_label_column] = result["y_codes"].map(
        effective_label_separator.join
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
                input_field_prefixes=cfg.input_field_prefixes,
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
        input_field_prefixes=default_cfg.input_field_prefixes,
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
