import re
from typing import Any, Mapping, Sequence, cast

import pandas as pd

from codllm.settings.options import SUPPORTED_TRAINING_INPUTS
from codllm.settings.schema import Config
from codllm.settings.types import TrainingInput
from codllm.input.mappings import DatasetMapping

UNKNOWN_VALUE = "unknown"
MISSING_VALUE_MARKERS: frozenset[str] = frozenset({"nan", "<na>", "none", "null"})
ICD10H_PREFIX_PATTERN = re.compile(r"^[A-Z]\d{2}$")
ICD10H_CANONICAL_PATTERN = re.compile(r"^[A-Z]\d{2}\.\d{3}$")


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
    supported_inputs = SUPPORTED_TRAINING_INPUTS
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
    field_separator: str | None = None,
) -> str:
    """Build text input from configured training input fields."""
    normalized_training_input = _normalize_training_input(training_input)
    effective_field_separator = (
        Config().text_field_separator if field_separator is None else field_separator
    )
    parts: list[str] = []
    for feature in normalized_training_input:
        if feature == "cod":
            cod_text = _get(row, mapping.text_col) or UNKNOWN_VALUE
            cod_text = re.sub(r"(?<=\w)\.(?=\w)", " ", cod_text)
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
    return effective_field_separator.join(parts)


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


def _build_label(codes: list[str], separator: str | None = None) -> str:
    """Convert code labels into a single seq2seq target string."""
    effective_separator = Config().label_separator if separator is None else separator
    return effective_separator.join(codes)


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
                values = [part.strip().strip("'\"") for part in payload.split(",")]
        elif separator and separator in stripped:
            values = [part.strip() for part in stripped.split(separator)]
        else:
            values = [stripped]
    elif raw_codes is None or pd.isna(raw_codes):
        values = []
    else:
        values = [str(raw_codes)]
    return [code for code in (_normalize_code_value(item) for item in values) if code]
