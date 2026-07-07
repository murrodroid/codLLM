from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

import pandas as pd

from codllm.input.transform import (
    _build_label,
    _coerce_row_codes,
    _normalize_icd10h_code_shape,
)
from codllm.settings.schema import Config


@dataclass(frozen=True)
class LabelStandardizationRule:
    """One reviewed dataset-level label standardization rule."""

    rule_id: str
    target_codes: tuple[str, ...]
    source_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    text_regex: str | None = None
    old_codes: tuple[str, ...] = ()
    old_code_blocks: tuple[str, ...] = ()
    description: str | None = None
    rationale: str | None = None
    text_pattern: Pattern[str] | None = None


@dataclass(frozen=True)
class LabelStandardizationOverride:
    """One row-level label standardization override."""

    rule_id: str
    source_id: str
    record_id: str
    new_codes: tuple[str, ...]
    old_label: str | None = None
    rationale: str | None = None


def build_label_standardization_metadata(cfg: Config) -> dict[str, Any]:
    """Build cache metadata for label standardization overlay files."""
    rules_path = _resolve_standardization_path(cfg.label_standardization_rules_path)
    overrides_path = _resolve_standardization_path(
        cfg.label_standardization_overrides_path
    )
    return {
        "enabled": cfg.label_standardization_enabled,
        "rules_path": cfg.label_standardization_rules_path,
        "overrides_path": cfg.label_standardization_overrides_path,
        "rules_file": _file_signature(rules_path)
        if cfg.label_standardization_enabled
        else None,
        "overrides_file": _file_signature(overrides_path)
        if cfg.label_standardization_enabled
        else None,
    }


def apply_label_standardization(cfg: Config, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply reviewed label standardization rules to a processed dataframe."""
    if not cfg.label_standardization_enabled:
        return dataframe
    if dataframe.empty:
        return dataframe.reset_index(drop=True)
    if "y_codes" not in dataframe.columns:
        raise KeyError(
            "Processed dataframe must contain 'y_codes' for label standardization."
        )

    rules = _load_rules(cfg)
    overrides = _load_overrides(cfg)
    if not rules and not overrides:
        return dataframe.reset_index(drop=True)

    standardized = dataframe.reset_index(drop=True).copy()
    rows_changed = _apply_rules(cfg, standardized, rules)
    rows_changed += _apply_overrides(cfg, standardized, overrides)
    standardized.attrs["label_standardization"] = {
        "rules_loaded": len(rules),
        "overrides_loaded": len(overrides),
        "rows_changed": rows_changed,
    }
    return standardized.reset_index(drop=True)


def _resolve_standardization_path(raw_path: str) -> Path:
    """Resolve a label standardization path without anchoring it to raw data."""
    return Path(raw_path).expanduser()


def _file_signature(path: Path) -> dict[str, Any]:
    """Return a cache-safe file signature for a curation overlay file."""
    signature: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
    }
    if path.exists():
        stats = path.stat()
        signature["size_bytes"] = stats.st_size
        signature["mtime_ns"] = stats.st_mtime_ns
        signature["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return signature


def _load_rules(cfg: Config) -> list[LabelStandardizationRule]:
    """Load label standardization rules from the configured TOML file."""
    path = _resolve_standardization_path(cfg.label_standardization_rules_path)
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid label standardization TOML at '{path}'.") from exc

    raw_rules = payload.get("rules", [])
    if raw_rules is None:
        return []
    if not isinstance(raw_rules, list):
        raise ValueError("Label standardization TOML must use [[rules]] entries.")

    rules: list[LabelStandardizationRule] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"Label standardization rule {index} must be a table.")
        rules.append(_parse_rule(raw_rule, index))
    return rules


def _parse_rule(raw_rule: dict[str, Any], index: int) -> LabelStandardizationRule:
    """Parse and validate one TOML rule entry."""
    rule_id = _required_string(raw_rule.get("id"), f"rules[{index}].id")
    target_codes = _code_tuple(
        raw_rule.get("target_codes"), f"rules[{index}].target_codes", required=True
    )
    text_regex = _optional_string(
        raw_rule.get("text_regex"), f"rules[{index}].text_regex"
    )
    text_pattern = None
    if text_regex is not None:
        try:
            text_pattern = re.compile(text_regex)
        except re.error as exc:
            raise ValueError(
                f"Label standardization rule '{rule_id}' has an invalid text_regex."
            ) from exc

    rule = LabelStandardizationRule(
        rule_id=rule_id,
        target_codes=target_codes,
        source_ids=_string_tuple(
            raw_rule.get("source_ids"), f"rules[{index}].source_ids"
        ),
        record_ids=_string_tuple(
            raw_rule.get("record_ids"), f"rules[{index}].record_ids"
        ),
        text_regex=text_regex,
        old_codes=_code_tuple(raw_rule.get("old_codes"), f"rules[{index}].old_codes"),
        old_code_blocks=_code_block_tuple(
            raw_rule.get("old_code_blocks"), f"rules[{index}].old_code_blocks"
        ),
        description=_optional_string(
            raw_rule.get("description"), f"rules[{index}].description"
        ),
        rationale=_optional_string(
            raw_rule.get("rationale"), f"rules[{index}].rationale"
        ),
        text_pattern=text_pattern,
    )
    if not _has_selector(rule):
        raise ValueError(
            f"Label standardization rule '{rule_id}' must define at least one selector."
        )
    return rule


def _load_overrides(cfg: Config) -> list[LabelStandardizationOverride]:
    """Load row-level label standardization overrides from the configured CSV file."""
    path = _resolve_standardization_path(cfg.label_standardization_overrides_path)
    if not path.exists():
        return []
    try:
        overrides_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return []

    required_columns = {
        "rule_id",
        "source_id",
        "record_id",
        "old_label",
        "new_label",
        "rationale",
    }
    missing_columns = required_columns.difference(overrides_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            "Label standardization overrides CSV is missing required columns: "
            f"{missing}."
        )

    overrides: list[LabelStandardizationOverride] = []
    seen_keys: set[tuple[str, str]] = set()
    for index, row in overrides_df.iterrows():
        row_number = int(index) + 2
        source_id = _required_string(
            row["source_id"], f"overrides row {row_number}.source_id"
        )
        record_id = _required_string(
            row["record_id"], f"overrides row {row_number}.record_id"
        )
        key = (source_id, record_id)
        if key in seen_keys:
            raise ValueError(
                "Label standardization overrides contain duplicate source_id/record_id "
                f"for '{source_id}'/'{record_id}'."
            )
        seen_keys.add(key)
        overrides.append(
            LabelStandardizationOverride(
                rule_id=_optional_string(
                    row["rule_id"], f"overrides row {row_number}.rule_id"
                )
                or "manual_override",
                source_id=source_id,
                record_id=record_id,
                old_label=_optional_string(
                    row["old_label"], f"overrides row {row_number}.old_label"
                ),
                new_codes=_parse_label_codes(
                    row["new_label"], f"overrides row {row_number}.new_label", cfg
                ),
                rationale=_optional_string(
                    row["rationale"], f"overrides row {row_number}.rationale"
                ),
            )
        )
    return overrides


def _apply_rules(
    cfg: Config,
    dataframe: pd.DataFrame,
    rules: list[LabelStandardizationRule],
) -> int:
    """Apply the first matching dataset-level rule to each row."""
    rows_changed = 0
    for index, row in dataframe.iterrows():
        current_codes = _row_codes(row, cfg)
        for rule in rules:
            if not _rule_matches(row, cfg, current_codes, rule):
                continue
            if tuple(current_codes) != rule.target_codes:
                _set_row_codes(dataframe, int(index), rule.target_codes, cfg)
                rows_changed += 1
            break
    return rows_changed


def _apply_overrides(
    cfg: Config,
    dataframe: pd.DataFrame,
    overrides: list[LabelStandardizationOverride],
) -> int:
    """Apply exact row-level overrides after dataset-level rules."""
    overrides_by_key = {
        (override.source_id, override.record_id): override for override in overrides
    }
    rows_changed = 0
    for index, row in dataframe.iterrows():
        key = (_row_text(row, "source_id"), _row_text(row, "record_id"))
        override = overrides_by_key.get(key)
        if override is None:
            continue
        current_label = _build_label(
            _row_codes(row, cfg), separator=cfg.label_separator
        )
        if override.old_label is not None and override.old_label != current_label:
            raise ValueError(
                "Label standardization override old_label mismatch for "
                f"'{override.source_id}'/'{override.record_id}': expected "
                f"'{override.old_label}', found '{current_label}'."
            )
        if tuple(_row_codes(row, cfg)) != override.new_codes:
            _set_row_codes(dataframe, int(index), override.new_codes, cfg)
            rows_changed += 1
    return rows_changed


def _rule_matches(
    row: pd.Series,
    cfg: Config,
    current_codes: list[str],
    rule: LabelStandardizationRule,
) -> bool:
    """Return whether one processed row matches a standardization rule."""
    if rule.source_ids and _row_text(row, "source_id") not in rule.source_ids:
        return False
    if rule.record_ids and _row_text(row, "record_id") not in rule.record_ids:
        return False
    if rule.text_pattern is not None:
        text = _row_cod_text(row, cfg)
        if rule.text_pattern.search(text) is None:
            return False
    if rule.old_codes and not set(current_codes).intersection(rule.old_codes):
        return False
    if rule.old_code_blocks:
        current_blocks = {code[:3] for code in current_codes}
        if not current_blocks.intersection(rule.old_code_blocks):
            return False
    return True


def _row_codes(row: pd.Series, cfg: Config) -> list[str]:
    """Return normalized codes from one processed row."""
    return [
        _normalize_icd10h_code_shape(code)
        for code in _coerce_row_codes(row["y_codes"], cfg.label_separator)
        if _normalize_icd10h_code_shape(code)
    ]


def _row_text(row: pd.Series, column: str) -> str:
    """Return a string value from one processed row."""
    value = row.get(column)
    if value is None:
        return ""
    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return ""
    return str(value).strip()


def _row_cod_text(row: pd.Series, cfg: Config) -> str:
    """Return the COD segment from one processed row text value."""
    text = _row_text(row, cfg.dataset_text_column)
    cod_prefix = cfg.input_field_prefixes.get("cod", "")
    if not cod_prefix:
        return text
    if list(cfg.training_input) == ["cod"] and not text.startswith(cod_prefix):
        return text

    segments = (
        text.split(cfg.text_field_separator) if cfg.text_field_separator else [text]
    )
    for segment in segments:
        stripped_segment = segment.strip()
        if stripped_segment.startswith(cod_prefix):
            return stripped_segment[len(cod_prefix) :].strip()
    if text.startswith(cod_prefix):
        return text[len(cod_prefix) :].strip()
    return text


def _set_row_codes(
    dataframe: pd.DataFrame,
    index: int,
    codes: tuple[str, ...],
    cfg: Config,
) -> None:
    """Set both structured and string label columns for one processed row."""
    updated_codes = list(codes)
    dataframe.at[index, "y_codes"] = updated_codes
    dataframe.at[index, cfg.dataset_label_column] = _build_label(
        updated_codes,
        separator=cfg.label_separator,
    )


def _has_selector(rule: LabelStandardizationRule) -> bool:
    """Return whether a rule has at least one matching selector."""
    return any(
        (
            rule.source_ids,
            rule.record_ids,
            rule.text_regex,
            rule.old_codes,
            rule.old_code_blocks,
        )
    )


def _required_string(value: Any, field_name: str) -> str:
    """Parse a required non-empty string value."""
    parsed = _optional_string(value, field_name)
    if parsed is None:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return parsed


def _optional_string(value: Any, field_name: str) -> str | None:
    """Parse an optional non-empty string value."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    stripped = value.strip()
    return stripped or None


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """Parse an optional string-or-string-list value."""
    return _tuple_from_value(value, field_name, normalize_codes=False)


def _code_tuple(
    value: Any,
    field_name: str,
    required: bool = False,
) -> tuple[str, ...]:
    """Parse an optional ICD10h code-or-code-list value."""
    parsed = _tuple_from_value(value, field_name, normalize_codes=True)
    if required and not parsed:
        raise ValueError(f"{field_name} must contain at least one code.")
    return parsed


def _code_block_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """Parse an optional ICD10h chapter-block code list."""
    parsed = _tuple_from_value(value, field_name, normalize_codes=True)
    blocks: list[str] = []
    for code in parsed:
        block = code[:3]
        if len(block) != 3:
            raise ValueError(f"{field_name} contains invalid code block '{code}'.")
        if block not in blocks:
            blocks.append(block)
    return tuple(blocks)


def _tuple_from_value(
    value: Any,
    field_name: str,
    normalize_codes: bool,
) -> tuple[str, ...]:
    """Parse an optional scalar or list value into a unique tuple."""
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a string or a list of strings.")

    parsed: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings.")
        stripped = item.strip()
        if not stripped:
            continue
        normalized = (
            _normalize_icd10h_code_shape(stripped) if normalize_codes else stripped
        )
        if normalized and normalized not in parsed:
            parsed.append(normalized)
    return tuple(parsed)


def _parse_label_codes(value: Any, field_name: str, cfg: Config) -> tuple[str, ...]:
    """Parse and validate one override label value."""
    parsed: list[str] = []
    for code in _coerce_row_codes(value, cfg.label_separator):
        normalized = _normalize_icd10h_code_shape(code)
        if normalized and normalized not in parsed:
            parsed.append(normalized)
    if not parsed:
        raise ValueError(f"{field_name} must contain at least one ICD10h label.")
    if len(parsed) > cfg.max_label_count:
        raise ValueError(
            f"{field_name} contains {len(parsed)} labels, but max_label_count is "
            f"{cfg.max_label_count}."
        )
    return tuple(parsed)
