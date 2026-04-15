"""Tests for ICD10h Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from codllm.labels.schemas import ICD10hCode, ICD10hCodeList


# --- ICD10hCode ---


def test_valid_code() -> None:
    c = ICD10hCode(code="A00.000")
    assert c.code == "A00.000"


def test_valid_code_strips_whitespace() -> None:
    c = ICD10hCode(code="  B12.345  ")
    assert c.code == "B12.345"


@pytest.mark.parametrize(
    "bad",
    ["bad", "a00.000", "A0.000", "A00.00", "A00.0000", ""],
)
def test_invalid_code_raises(bad: str) -> None:
    with pytest.raises(ValidationError):
        ICD10hCode(code=bad)


# --- ICD10hCodeList ---


def test_from_raw_string_single() -> None:
    cl = ICD10hCodeList.from_raw_string("A00.000")
    assert len(cl.codes) == 1
    assert cl.codes[0].code == "A00.000"


def test_from_raw_string_multiple() -> None:
    cl = ICD10hCodeList.from_raw_string("A00.000 | A01.005")
    assert len(cl.codes) == 2


def test_from_raw_string_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        ICD10hCodeList.from_raw_string("GARBAGE")


def test_to_label_string_roundtrip() -> None:
    raw = "A00.000 | B12.345"
    cl = ICD10hCodeList.from_raw_string(raw)
    assert cl.to_label_string() == raw


def test_empty_string_raises() -> None:
    with pytest.raises(ValidationError):
        ICD10hCodeList.from_raw_string("")


def test_min_length_enforced() -> None:
    with pytest.raises(ValidationError):
        ICD10hCodeList(codes=[])
