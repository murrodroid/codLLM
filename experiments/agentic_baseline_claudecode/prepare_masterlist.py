"""Flatten the ICD-10h masterlist xlsx into a grep-friendly tab-separated file.

Writes every Masterlist column (with a header row) so the agent can use them
as case-by-case hints rather than hard filters.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "data" / "raw" / "ICD10h_Masterlist_2024.xlsx"
DST = PROJECT_ROOT / "data" / "raw" / "ICD10h_Masterlist_2024.tsv"

CODE_RE = re.compile(r"^[A-Z]\d{2}\.\d{3}$")

COLUMNS = [
    "IDMasterlist",
    "ICD10h",
    "ICD10",
    "icd10_2levelCATEGORY",
    "ICD10_2levelCAUSE",
    "ICD10h_DESCRIPTION",
    "HistCat",
    "DoNotUse",
    "NotForUnderlying",
    "GenderSpecific",
]


def _clean_cell(value: object) -> str:
    """Strip whitespace and replace tabs/newlines so each row stays single-line."""
    if value is None:
        return ""
    text = str(value).strip()
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def main() -> None:
    df = pd.read_excel(SRC, sheet_name="Masterlist", engine="openpyxl", dtype=str)
    df = df[df["ICD10h"].str.match(CODE_RE, na=False)].copy()

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Masterlist is missing expected columns: {missing}")

    df = df[COLUMNS].fillna("")
    header = "\t".join(COLUMNS)
    body_rows = [
        "\t".join(_clean_cell(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    DST.write_text("\n".join([header, *body_rows]) + "\n", encoding="utf-8")
    print(f"Wrote {len(body_rows)} entries (+ header) to {DST}")


if __name__ == "__main__":
    main()
