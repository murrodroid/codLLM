from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, cast

import pandas as pd

from codllm.config import Config, DataSourceConfig, TrainingInput
from codllm.data_augmentation import (
    accent_random_vowel,
    delete_random_char,
    insert_random_whitespace,
    qwerty_misspell,
    swap_adjacent_chars,
)
from codllm.path_utils import resolve_source_path

UNKNOWN_VALUE = "unknown"
MISSING_VALUE_MARKERS: frozenset[str] = frozenset({"nan", "<na>", "none", "null"})
ICD10H_PREFIX_PATTERN = re.compile(r"^[A-Z]\d{2}$")
ICD10H_CANONICAL_PATTERN = re.compile(r"^[A-Z]\d{2}\.\d{3}$")


@dataclass
class DatasetMapping:
    """Describe how to map one source dataset into the canonical training schema."""

    text_col: int
    single_code_col: int
    multi_code_cols: list[int] = field(default_factory=list)
    sex_col: int | None = None
    sex_map: dict[str, str] = field(default_factory=dict)
    age_col: int | None = None
    record_id_col: int | None = None
    skip_rows: list[int] = field(default_factory=list)


BELGIUM_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=11,
    multi_code_cols=[12, 13, 14, 15, 16],
    sex_col=2,
    sex_map={"1": "male", "2": "female"},
    age_col=3,
    record_id_col=0,
    skip_rows=[0],
)

AMSTERDAM_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=7,
    multi_code_cols=[7, 9, 11, 13, 15, 17],
    sex_col=2,
    sex_map={
        "man": "male",
        "vrouw": "female",
        "m": "male",
        "v": "female",
        "1": "male",
        "2": "female",
    },
    age_col=3,
    record_id_col=0,
)

COPENHAGEN_MAPPING = DatasetMapping(
    text_col=37,
    single_code_col=39,
    multi_code_cols=[],
    sex_col=23,
    sex_map={"Mand": "male", "Kvinde": "female"},
    age_col=12,
    record_id_col=0,
    skip_rows=[0],
)

MASTERLIST_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=1,
    multi_code_cols=[],
    sex_col=None,
    sex_map={},
    age_col=None,
    record_id_col=0,
)

MAPPING_REGISTRY: dict[str, DatasetMapping] = {
    "belgium": BELGIUM_MAPPING,
    "amsterdam": AMSTERDAM_MAPPING,
    "copenhagen": COPENHAGEN_MAPPING,
    "masterlist": MASTERLIST_MAPPING,
}

PERTURBATION_REGISTRY: dict[str, Any] = {
    "swap_adjacent_chars": swap_adjacent_chars,
    "delete_random_char": delete_random_char,
    "insert_random_whitespace": insert_random_whitespace,
    "accent_random_vowel": accent_random_vowel,
    "qwerty_misspell": qwerty_misspell,
}


def _processed_columns(text_column: str, label_column: str) -> list[str]:
    """Return canonical processed-data column order for configured text/label names."""
    return [
        "source_id",
        "record_id",
        "source_path",
        text_column,
        "y_codes",
        label_column,
    ]


def _normalize_training_input(training_input: Sequence[str]) -> list[TrainingInput]:
    """Validate and normalize requested training input fields."""
    normalized: list[TrainingInput] = []
    supported_inputs = Config.SUPPORTED_TRAINING_INPUTS
    for feature in training_input:
        cleaned = feature.strip().lower()
        if cleaned not in supported_inputs:
            supported = ", ".join(supported_inputs)
            raise ValueError(
                f"Unsupported training input '{feature}'. Supported values are: {supported}."
            )
        normalized_feature = cast(TrainingInput, cleaned)
        if normalized_feature not in normalized:
            normalized.append(normalized_feature)
    if not normalized:
        raise ValueError("training_input must contain at least one field.")
    return normalized


def _get(row: pd.Series, col: int) -> str | None:
    """Get a value from a row by positional index."""
    if col < 0 or col >= len(row):
        return None
    value = row.iloc[col]
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in MISSING_VALUE_MARKERS:
        return None
    return text


def _format_age(raw_age: str | None) -> str:
    """Normalize age into a compact numeric string."""
    if raw_age is None:
        return UNKNOWN_VALUE
    try:
        numeric_age = round(float(raw_age), 2)
    except ValueError:
        return UNKNOWN_VALUE
    if numeric_age.is_integer():
        return str(int(numeric_age))
    return f"{numeric_age:.2f}".rstrip("0").rstrip(".")


def _format_sex(raw_sex: str | None, sex_map: Mapping[str, str]) -> str:
    """Map raw sex values into canonical values."""
    if raw_sex is None:
        return UNKNOWN_VALUE
    raw = raw_sex.strip()
    normalized_raw = raw.lower()
    if raw in sex_map:
        return sex_map[raw]
    if normalized_raw in sex_map:
        return sex_map[normalized_raw]
    return UNKNOWN_VALUE


def _build_text(
    row: pd.Series,
    mapping: DatasetMapping,
    training_input: Sequence[str],
    field_separator: str = Config.DEFAULT_TEXT_FIELD_SEPARATOR,
) -> str:
    """Build text input from configured training input fields."""
    normalized_training_input = _normalize_training_input(training_input)
    parts: list[str] = []
    for feature in normalized_training_input:
        if feature == "cod":
            cod_text = _get(row, mapping.text_col) or UNKNOWN_VALUE
            parts.append(f"cod: {cod_text}")
        elif feature == "age":
            raw_age = (
                _get(row, mapping.age_col) if mapping.age_col is not None else None
            )
            parts.append(f"age: {_format_age(raw_age)}")
        else:
            raw_sex = (
                _get(row, mapping.sex_col) if mapping.sex_col is not None else None
            )
            parts.append(f"sex: {_format_sex(raw_sex, mapping.sex_map)}")
    return field_separator.join(parts)


def _collect_codes(row: pd.Series, mapping: DatasetMapping) -> list[str]:
    """Collect code values from multi-code columns first, then single-code fallback."""
    codes: list[str] = []
    seen_codes: set[str] = set()
    for col in mapping.multi_code_cols or []:
        code = _get(row, col)
        if code and code not in seen_codes:
            codes.append(code)
            seen_codes.add(code)
    if not codes:
        single_code = _get(row, mapping.single_code_col)
        if single_code:
            codes.append(single_code)
    return codes


def _build_y(row: pd.Series, mapping: DatasetMapping) -> list[str]:
    """Build complete target code list for one source row."""
    return _collect_codes(row, mapping)


def _build_label(
    codes: list[str], separator: str = Config.DEFAULT_LABEL_SEPARATOR
) -> str:
    """Convert code labels into a single seq2seq target string."""
    return separator.join(codes)


def _normalize_code_value(value: Any) -> str:
    """Normalize one code token to uppercase text."""
    if value is None:
        return ""
    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return ""
    code = str(value).strip().upper()
    if not code or code.lower() in MISSING_VALUE_MARKERS:
        return ""
    return code


def _normalize_icd10h_code_shape(code: str) -> str:
    """Normalize one ICD10h code candidate into canonical dot+3-digit shape."""
    normalized = _normalize_code_value(code)
    if not normalized:
        return ""
    if ICD10H_CANONICAL_PATTERN.fullmatch(normalized):
        return normalized
    if normalized.count(".") != 1:
        return normalized

    prefix, suffix = normalized.split(".", 1)
    if not ICD10H_PREFIX_PATTERN.fullmatch(prefix):
        return normalized
    if not suffix.isdigit():
        return normalized
    if len(suffix) < 1 or len(suffix) > 3:
        return normalized
    return f"{prefix}.{suffix.ljust(3, '0')}"


def _coerce_row_codes(raw_codes: Any, separator: str) -> list[str]:
    """Convert one row y_codes payload into a normalized code list."""
    if isinstance(raw_codes, list):
        values = raw_codes
    elif isinstance(raw_codes, tuple):
        values = list(raw_codes)
    elif hasattr(raw_codes, "tolist") and not isinstance(raw_codes, (str, bytes)):
        converted = raw_codes.tolist()
        if isinstance(converted, list):
            values = converted
        else:
            values = [converted]
    elif isinstance(raw_codes, str):
        stripped = raw_codes.strip()
        if stripped == "":
            values = []
        elif stripped.startswith("[") and stripped.endswith("]"):
            payload = stripped[1:-1].strip()
            if payload == "":
                values = []
            else:
                values = [
                    part.strip().strip("'\"") for part in payload.split(",")
                ]
        elif separator and separator in stripped:
            values = [part.strip() for part in stripped.split(separator)]
        else:
            values = [stripped]
    elif raw_codes is None or pd.isna(raw_codes):
        values = []
    else:
        values = [str(raw_codes)]
    return [code for code in (_normalize_code_value(item) for item in values) if code]


def _resolve_label_reference_workbook_path(cfg: Config) -> Path:
    """Resolve the ICD10h reference workbook used for label harmonization."""
    reference_path = Path(cfg.pretrain_masterlist_path)
    if reference_path.is_absolute():
        return reference_path

    data_root = Path(cfg.data_raw_dir)
    if reference_path.parts and data_root.parts:
        if reference_path.parts[0] == data_root.parts[-1]:
            trimmed_candidate = data_root / Path(*reference_path.parts[1:])
            if trimmed_candidate.exists():
                return trimmed_candidate

    return resolve_source_path(cfg.pretrain_masterlist_path, cfg.data_raw_dir)


def _load_label_reference_tables(cfg: Config) -> tuple[set[str], dict[str, str]]:
    """Load masterlist-valid labels and 2020->2024 transfer mapping."""
    workbook_path = _resolve_label_reference_workbook_path(cfg)
    if not workbook_path.exists():
        raise FileNotFoundError(
            "Label harmonization is enabled, but masterlist workbook was not found at "
            f"'{workbook_path}'."
        )

    masterlist_df = pd.read_excel(
        workbook_path,
        sheet_name=cfg.pretrain_masterlist_sheet_name,
        dtype=str,
    )
    if "ICD10h" not in masterlist_df.columns:
        raise KeyError(
            "Masterlist sheet must contain an 'ICD10h' column for label harmonization."
        )

    master_codes = {
        _normalize_icd10h_code_shape(code)
        for code in masterlist_df["ICD10h"].tolist()
        if _normalize_icd10h_code_shape(code)
    }

    try:
        transfer_df = pd.read_excel(
            workbook_path,
            sheet_name=cfg.pretrain_transfer_sheet_name,
            dtype=str,
        )
    except ValueError:
        transfer_df = pd.DataFrame(columns=["ICD10h_oct2020", "ICD10h2024"])

    if transfer_df.empty:
        return master_codes, {}

    required_columns = {"ICD10h_oct2020", "ICD10h2024"}
    missing_columns = required_columns.difference(transfer_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            "Transfer sheet is missing required columns for label harmonization: "
            f"{missing}."
        )

    transfer_pairs = transfer_df[["ICD10h_oct2020", "ICD10h2024"]].copy()
    transfer_pairs["ICD10h_oct2020"] = transfer_pairs["ICD10h_oct2020"].apply(
        _normalize_icd10h_code_shape
    )
    transfer_pairs["ICD10h2024"] = transfer_pairs["ICD10h2024"].apply(
        _normalize_icd10h_code_shape
    )
    transfer_pairs = transfer_pairs[
        (transfer_pairs["ICD10h_oct2020"] != "")
        & (transfer_pairs["ICD10h2024"] != "")
    ]

    transfer_map: dict[str, str] = {}
    for old_code, grouped in transfer_pairs.groupby("ICD10h_oct2020"):
        targets = sorted(set(grouped["ICD10h2024"].tolist()))
        if len(targets) > 1:
            rendered_targets = ", ".join(targets)
            raise ValueError(
                "Transfer sheet has ambiguous 2020->2024 mapping for "
                f"'{old_code}': {rendered_targets}."
            )
        transfer_map[old_code] = targets[0]

    return master_codes, transfer_map


def _map_code_with_transfer(code: str, transfer_map: Mapping[str, str]) -> str:
    """Map one label using transfer table and normalize shape."""
    normalized = _normalize_icd10h_code_shape(code)
    if not normalized:
        return ""
    mapped = transfer_map.get(normalized, normalized)
    mapped = _normalize_icd10h_code_shape(mapped)
    remapped = transfer_map.get(mapped, mapped)
    return _normalize_icd10h_code_shape(remapped)


def _harmonize_processed_labels(cfg: Config, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Map labels through transfer, normalize ICD10h shape, then drop unknown labels."""
    if dataframe.empty:
        return dataframe.reset_index(drop=True)

    master_codes, transfer_map = _load_label_reference_tables(cfg)
    if not master_codes:
        raise ValueError("Masterlist has no valid ICD10h codes for label harmonization.")

    harmonized_codes: list[list[str]] = []
    keep_mask: list[bool] = []
    for raw_codes in dataframe["y_codes"].tolist():
        row_codes = _coerce_row_codes(raw_codes, cfg.label_separator)
        mapped_codes = [_map_code_with_transfer(code, transfer_map) for code in row_codes]
        should_keep = bool(mapped_codes) and all(code in master_codes for code in mapped_codes)
        keep_mask.append(should_keep)
        harmonized_codes.append(mapped_codes)

    harmonized = dataframe.copy()
    harmonized["y_codes"] = harmonized_codes
    harmonized[cfg.dataset_label_column] = harmonized["y_codes"].apply(
        lambda codes: _build_label(codes, separator=cfg.label_separator)
    )
    return harmonized.loc[keep_mask].reset_index(drop=True)


def _detect_file_type(path: Path, source: DataSourceConfig) -> str:
    """Determine file type from source config or file extension."""
    if source.file_type is not None:
        return source.file_type.lower().lstrip(".")
    return path.suffix.lower().lstrip(".")


def _read_raw_dataframe(path: Path, source: DataSourceConfig) -> pd.DataFrame:
    """Read one raw source file into a dataframe."""
    file_type = _detect_file_type(path, source)
    if file_type == "csv":
        return pd.read_csv(path, header=source.header, dtype=str, sep=source.sep)
    if file_type in {"xlsx", "xls"}:
        return pd.read_excel(
            path, header=source.header, dtype=str, sheet_name=source.sheet_name
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
    label_separator: str = Config.DEFAULT_LABEL_SEPARATOR,
    text_field_separator: str = Config.DEFAULT_TEXT_FIELD_SEPARATOR,
    data_raw_dir: str = Config.DEFAULT_DATA_RAW_DIR,
    text_column: str = Config.DEFAULT_DATASET_TEXT_COLUMN,
    label_column: str = Config.DEFAULT_DATASET_LABEL_COLUMN,
    drop_missing_label: bool = True,
) -> pd.DataFrame:
    """Load and process one source dataset into the canonical schema."""
    if max_labels < 1:
        raise ValueError("max_labels must be at least 1.")
    normalized_training_input = _normalize_training_input(training_input)
    source_path = resolve_source_path(source.path, data_raw_dir)
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
            lambda row: _get(row, mapping.record_id_col), axis=1
        )
        result["record_id"] = [
            record_id if record_id is not None else f"{source.source_id}:{source_idx}"
            for source_idx, record_id in zip(
                source_row_indices, extracted_ids.tolist(), strict=True
            )
        ]
    result["source_path"] = [str(source_path)] * len(filtered_raw_df)
    result[text_column] = filtered_raw_df.apply(
        lambda row: _build_text(
            row,
            mapping,
            normalized_training_input,
            field_separator=text_field_separator,
        ),
        axis=1,
    )
    result["y_codes"] = filtered_y_codes
    result[label_column] = result["y_codes"].apply(
        lambda codes: _build_label(codes, separator=label_separator)
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
    source = DataSourceConfig(
        source_id="inline_source",
        path=path,
        mapping_id="inline_mapping",
        sep=sep,
    )
    processed = load_source_dataset(
        source=source,
        mapping=mapping,
        training_input=training_input or list(Config.SUPPORTED_TRAINING_INPUTS),
        max_labels=max_labels,
        data_raw_dir="",
        text_column=Config.DEFAULT_DATASET_TEXT_COLUMN,
        label_column=Config.DEFAULT_DATASET_LABEL_COLUMN,
        drop_missing_label=False,
    )
    return pd.DataFrame(
        {
            Config.DEFAULT_DATASET_TEXT_COLUMN: processed[
                Config.DEFAULT_DATASET_TEXT_COLUMN
            ],
            "y": processed["y_codes"],
        }
    )
