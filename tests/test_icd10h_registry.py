"""Tests for icd10h_registry module."""

from __future__ import annotations

from pathlib import Path

import pytest

from codllm.labels.registry import (
    ICD10H_PATTERN,
    coarse_code,
    get_valid_codes,
    icd10_code,
    is_valid_code,
    load_masterlist,
    set_valid_codes,
)

MASTERLIST_PATH = Path("data/raw/ICD10h_Masterlist_2024.xlsx")


# --- Pattern ---


@pytest.mark.parametrize(
    "code",
    ["A00.000", "Z99.123", "B12.345"],
)
def test_pattern_matches_valid(code: str) -> None:
    assert ICD10H_PATTERN.match(code)


@pytest.mark.parametrize(
    "code",
    ["a00.000", "A0.000", "A000.00", "A00.00", "A00.0000", "bad", ""],
)
def test_pattern_rejects_invalid(code: str) -> None:
    assert not ICD10H_PATTERN.match(code)


# --- coarse / icd10 helpers ---


def test_coarse_code() -> None:
    assert coarse_code("A00.000") == "A00"
    assert coarse_code("B12.345") == "B12"


def test_icd10_code() -> None:
    assert icd10_code("A00.000") == "A00.0"
    assert icd10_code("B12.345") == "B12.3"


# --- Registry get/set ---


def test_set_and_get_valid_codes() -> None:
    codes = frozenset({"A00.000", "B01.001"})
    set_valid_codes(codes)
    assert get_valid_codes() is codes


def test_is_valid_code() -> None:
    set_valid_codes(frozenset({"A00.000"}))
    assert is_valid_code("A00.000") is True
    assert is_valid_code("Z99.999") is False


# --- Masterlist loading (integration) ---


@pytest.mark.skipif(
    not MASTERLIST_PATH.exists(),
    reason="Masterlist Excel file not present",
)
def test_load_masterlist() -> None:
    codes = load_masterlist(MASTERLIST_PATH)
    assert len(codes) == 14_088
    assert "A00.000" in codes
