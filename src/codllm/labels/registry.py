"""Registry of valid ICD10h codes loaded from the official masterlist."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ICD10H_PATTERN = re.compile(r"^[A-Z]\d{2}\.\d{3}$")

_valid_codes: frozenset[str] = frozenset()


def load_masterlist(path: str | Path) -> frozenset[str]:
    """Load all valid ICD10h codes from the Excel masterlist.

    Reads the ``Masterlist`` sheet, ``ICD10h`` column.
    """
    df = pd.read_excel(path, sheet_name="Masterlist", engine="openpyxl")
    codes = df["ICD10h"].dropna().astype(str).str.strip()
    codes = codes[codes.str.match(ICD10H_PATTERN)]
    return frozenset(codes)


def set_valid_codes(codes: frozenset[str]) -> None:
    """Replace the module-level registry with *codes*."""
    global _valid_codes
    _valid_codes = codes


def get_valid_codes() -> frozenset[str]:
    """Return the current module-level registry."""
    return _valid_codes


def is_valid_code(code: str) -> bool:
    """Check whether *code* is in the loaded registry."""
    return code in _valid_codes


def coarse_code(full_code: str) -> str:
    """Extract coarse category: ``'A00.000'`` → ``'A00'``."""
    return full_code[0] + full_code[1] + full_code[2]


def icd10_code(full_code: str) -> str:
    """Convert ICD10h to ICD10: ``'A00.000'`` → ``'A00.0'``."""
    return full_code[:4] + full_code[4]
