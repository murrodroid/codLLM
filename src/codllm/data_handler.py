from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DatasetMapping:
    """Describes how to map columns from a specific dataset source into the unified X/y format.

    All column references are integer indices (0-based) so they work regardless of header names.
    """

    text_col: int
    single_code_col: int
    multi_code_cols: list[int] = field(default_factory=list)
    gender_col: int | None = None
    gender_map: dict[str, str] = field(default_factory=dict)
    year_col: int | None = None
    age_col: int | None = None
    skip_rows: list[int] = field(default_factory=list)


BELGIUM_MAPPING = DatasetMapping(
    text_col=4,
    single_code_col=11,
    multi_code_cols=[12, 13, 14, 15, 16],
    gender_col=2,
    gender_map={"1": "male", "2": "female"},
    year_col=1,
    age_col=3,
    skip_rows=[0],
)


def _get(row: pd.Series, col: int) -> str | None:
    """Get a value from a row by positional index, returning None for missing/empty values."""
    val = row.iloc[col]
    if pd.notna(val) and str(val).strip():
        return str(val).strip()
    return None


def _build_x(row: pd.Series, mapping: DatasetMapping) -> str:
    """Build the X input string from metadata and original cause-of-death text."""
    parts = []

    if mapping.gender_col is not None:
        raw = _get(row, mapping.gender_col)
        parts.append(f"gender: {mapping.gender_map.get(raw, 'unknown') if raw else 'unknown'}")
    else:
        parts.append("gender: unknown")

    if mapping.year_col is not None:
        raw = _get(row, mapping.year_col)
        parts.append(f"year: {int(float(raw)) if raw else 'unknown'}")
    else:
        parts.append("year: unknown")

    if mapping.age_col is not None:
        raw = _get(row, mapping.age_col)
        parts.append(f"age: {round(float(raw), 2) if raw else 'unknown'}")
    else:
        parts.append("age: unknown")

    text = _get(row, mapping.text_col)
    parts.append(f"text: {text if text else 'unknown'}")

    return " | ".join(parts)


def _build_y(row: pd.Series, mapping: DatasetMapping) -> list[str]:
    """Collect ICD codes from multi-cause columns, falling back to the single-cause column."""
    codes = []
    for col in mapping.multi_code_cols:
        val = _get(row, col)
        if val:
            codes.append(val)

    if not codes:
        val = _get(row, mapping.single_code_col)
        if val:
            codes.append(val)

    return codes


def load_dataset(path: str, mapping: DatasetMapping) -> pd.DataFrame:
    """Load a dataset file and return a DataFrame with columns 'X' and 'y'.

    Args:
        path: Path to the dataset file (.xlsx or .csv).
        mapping: A DatasetMapping describing the column layout of this source.

    Returns:
        A DataFrame with 'X' (str) and 'y' (list[str]) columns.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path, header=0, dtype=str)
    else:
        df = pd.read_excel(path, header=0, dtype=str)

    if mapping.skip_rows:
        df = df.drop(index=mapping.skip_rows).reset_index(drop=True)

    result = pd.DataFrame()
    result["X"] = df.apply(lambda row: _build_x(row, mapping), axis=1)
    result["y"] = df.apply(lambda row: _build_y(row, mapping), axis=1)

    return result
