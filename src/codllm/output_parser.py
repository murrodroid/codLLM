"""Parse raw model output into validated ICD10h structured results."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from codllm.icd10h_registry import is_valid_code
from codllm.schemas import ICD10hCode, ICD10hCodeList


@dataclass
class ParseResult:
    """Result of parsing a single model output string.

    *parsed* is ``None`` when every token was invalid.
    *invalid_codes* captures tokens that failed validation so they can be
    inspected during evaluation — nothing is silently dropped.
    """

    raw_text: str
    parsed: ICD10hCodeList | None = None
    invalid_codes: list[str] = field(default_factory=list)


def parse_model_output(
    raw_text: str,
    separator: str = " | ",
    validate_registry: bool = False,
) -> ParseResult:
    """Parse a single raw output string.

    Parameters
    ----------
    raw_text:
        The raw string produced by the model.
    separator:
        Delimiter between codes in *raw_text*.
    validate_registry:
        If ``True``, also reject codes that pass the regex but are absent
        from the loaded :mod:`icd10h_registry`.
    """
    tokens = [t.strip() for t in raw_text.split(separator) if t.strip()]
    valid: list[ICD10hCode] = []
    invalid: list[str] = []

    for token in tokens:
        try:
            code = ICD10hCode(code=token)
        except ValidationError:
            invalid.append(token)
            continue

        if validate_registry and not is_valid_code(code.code):
            invalid.append(token)
            continue

        valid.append(code)

    parsed = ICD10hCodeList(codes=valid) if valid else None
    return ParseResult(raw_text=raw_text, parsed=parsed, invalid_codes=invalid)


def parse_batch_outputs(
    raw_texts: list[str],
    separator: str = " | ",
    validate_registry: bool = False,
) -> list[ParseResult]:
    """Parse a batch of raw output strings."""
    return [
        parse_model_output(rt, separator=separator, validate_registry=validate_registry)
        for rt in raw_texts
    ]
