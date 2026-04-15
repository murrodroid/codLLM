"""Decode and validate inference outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from codllm.icd10h_registry import is_valid_code, load_masterlist, set_valid_codes
from codllm.schemas import ICD10hCode, ICD10hCodeList


@dataclass
class ParseResult:
    """Result of parsing one raw model output string."""

    raw_text: str
    parsed: ICD10hCodeList | None = None
    invalid_codes: list[str] = field(default_factory=list)


def parse_model_output(
    raw_text: str,
    separator: str = " | ",
    validate_registry: bool = False,
) -> ParseResult:
    """Parse one raw output string into validated ICD10h codes."""
    tokens = [token.strip() for token in raw_text.split(separator) if token.strip()]
    valid_codes: list[ICD10hCode] = []
    invalid_codes: list[str] = []

    for token in tokens:
        try:
            code = ICD10hCode(code=token)
        except ValidationError:
            invalid_codes.append(token)
            continue

        if validate_registry and not is_valid_code(code.code):
            invalid_codes.append(token)
            continue
        valid_codes.append(code)

    parsed = ICD10hCodeList(codes=valid_codes) if valid_codes else None
    return ParseResult(raw_text=raw_text, parsed=parsed, invalid_codes=invalid_codes)


def parse_batch_outputs(
    raw_texts: list[str],
    separator: str = " | ",
    validate_registry: bool = False,
) -> list[ParseResult]:
    """Parse a batch of raw model outputs."""
    return [
        parse_model_output(
            raw_text=raw_text,
            separator=separator,
            validate_registry=validate_registry,
        )
        for raw_text in raw_texts
    ]


def configure_registry_validation(masterlist_path: str) -> None:
    """Load the ICD10h masterlist for registry-aware inference validation."""
    set_valid_codes(load_masterlist(masterlist_path))


def build_prediction_dataframe(
    inputs: pd.DataFrame,
    raw_predictions: list[str],
    *,
    label_separator: str,
    validate_registry: bool = False,
) -> pd.DataFrame:
    """Append parsed prediction columns to an inference input dataframe."""
    parsed_predictions = parse_batch_outputs(
        raw_texts=raw_predictions,
        separator=label_separator,
        validate_registry=validate_registry,
    )
    result = inputs.copy()
    result["prediction_raw_text"] = [item.raw_text for item in parsed_predictions]
    result["prediction_label"] = [
        item.parsed.to_label_string(separator=label_separator)
        if item.parsed is not None
        else ""
        for item in parsed_predictions
    ]
    result["prediction_code_count"] = [
        len(item.parsed.codes) if item.parsed is not None else 0
        for item in parsed_predictions
    ]
    result["prediction_invalid_codes_json"] = [
        json.dumps(item.invalid_codes) for item in parsed_predictions
    ]
    result["prediction_invalid_code_count"] = [
        len(item.invalid_codes) for item in parsed_predictions
    ]
    return result.reset_index(drop=True)
