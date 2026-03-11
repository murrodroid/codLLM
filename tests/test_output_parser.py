"""Tests for the output parser."""

from __future__ import annotations

from codllm.icd10h_registry import set_valid_codes
from codllm.output_parser import parse_batch_outputs, parse_model_output


# --- parse_model_output ---


def test_all_valid() -> None:
    result = parse_model_output("A00.000 | A01.005")
    assert result.parsed is not None
    assert len(result.parsed.codes) == 2
    assert result.invalid_codes == []


def test_mixed_valid_and_invalid() -> None:
    result = parse_model_output("A00.000 | GARBAGE")
    assert result.parsed is not None
    assert len(result.parsed.codes) == 1
    assert result.parsed.codes[0].code == "A00.000"
    assert result.invalid_codes == ["GARBAGE"]


def test_all_invalid() -> None:
    result = parse_model_output("GARBAGE | JUNK")
    assert result.parsed is None
    assert result.invalid_codes == ["GARBAGE", "JUNK"]


def test_raw_text_preserved() -> None:
    raw = "A00.000 | BAD"
    result = parse_model_output(raw)
    assert result.raw_text == raw


def test_custom_separator() -> None:
    result = parse_model_output("A00.000,B01.001", separator=",")
    assert result.parsed is not None
    assert len(result.parsed.codes) == 2


# --- Registry validation ---


def test_validate_registry_rejects_unknown() -> None:
    set_valid_codes(frozenset({"A00.000"}))
    result = parse_model_output("A00.000 | B99.999", validate_registry=True)
    assert result.parsed is not None
    assert len(result.parsed.codes) == 1
    assert result.parsed.codes[0].code == "A00.000"
    assert result.invalid_codes == ["B99.999"]


# --- Batch ---


def test_parse_batch_outputs() -> None:
    results = parse_batch_outputs(["A00.000", "BAD"])
    assert len(results) == 2
    assert results[0].parsed is not None
    assert results[1].parsed is None
