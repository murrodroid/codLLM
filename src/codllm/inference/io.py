from __future__ import annotations

from pathlib import Path

import pandas as pd

RECORD_ID_COLUMN = "record_id"


def _read_text_lines(path: Path, text_column: str) -> pd.DataFrame:
    """Read one plain-text input file as one inference row per line."""
    lines = path.read_text().splitlines()
    return pd.DataFrame({text_column: lines})


def _read_structured_input(path: Path) -> pd.DataFrame:
    """Read one structured inference input file into a dataframe."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(
        "Unsupported inference input format. Use .csv, .tsv, .parquet, .jsonl, or .txt."
    )


def _ensure_record_id_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure inference rows have a stable record identifier column."""
    result = dataframe.copy()
    if RECORD_ID_COLUMN in result.columns:
        record_ids = result[RECORD_ID_COLUMN].fillna("").astype(str).str.strip()
        missing_mask = record_ids.eq("")
        if missing_mask.any():
            replacements = [f"row-{idx:06d}" for idx in result.index[missing_mask]]
            result.loc[missing_mask, RECORD_ID_COLUMN] = replacements
        return result

    result.insert(0, RECORD_ID_COLUMN, [f"row-{idx:06d}" for idx in range(len(result))])
    return result


def load_inference_inputs(path: str | Path, text_column: str) -> pd.DataFrame:
    """Load inference input rows into a canonical dataframe."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Inference input file was not found: '{input_path}'.")

    if input_path.suffix.lower() == ".txt":
        dataframe = _read_text_lines(input_path, text_column=text_column)
    else:
        dataframe = _read_structured_input(input_path)

    if text_column not in dataframe.columns:
        raise KeyError(
            f"Inference input is missing configured text column '{text_column}'."
        )
    if dataframe.empty:
        raise ValueError("Inference input is empty.")

    result = _ensure_record_id_column(dataframe)
    result[text_column] = result[text_column].fillna("").astype(str)
    return result.reset_index(drop=True)


def write_inference_outputs(dataframe: pd.DataFrame, path: str | Path) -> None:
    """Write inference outputs using an extension-driven file format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
        return
    if suffix == ".tsv":
        dataframe.to_csv(output_path, index=False, sep="\t")
        return
    if suffix == ".parquet":
        dataframe.to_parquet(output_path, index=False)
        return
    if suffix == ".jsonl":
        dataframe.to_json(output_path, orient="records", lines=True)
        return
    raise ValueError(
        "Unsupported inference output format. Use .csv, .tsv, .parquet, or .jsonl."
    )
