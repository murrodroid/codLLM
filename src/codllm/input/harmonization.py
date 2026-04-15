from pathlib import Path
from typing import Mapping

import pandas as pd

from codllm.runtime.paths import resolve_source_path
from codllm.settings.schema import Config
from codllm.input.transform import (
    _build_label,
    _coerce_row_codes,
    _normalize_icd10h_code_shape,
)


def _resolve_label_reference_workbook_path(cfg: Config) -> Path:
    """Resolve the ICD10h reference workbook used for label harmonization."""
    reference_path = Path(cfg.pretrain_masterlist_path)
    if reference_path.is_absolute():
        return reference_path

    data_root = Path(cfg.data_raw_dir)
    if reference_path.parts and data_root.parts:
        if reference_path.parts[0] == data_root.parts[-1]:
            trimmed_candidate = data_root / Path(*reference_path.parts[1:])
            if trimmed_candidate.exists():
                return trimmed_candidate

    return resolve_source_path(cfg.pretrain_masterlist_path, cfg.data_raw_dir)


def _load_label_reference_tables(cfg: Config) -> tuple[set[str], dict[str, str]]:
    """Load masterlist-valid labels and 2020->2024 transfer mapping."""
    workbook_path = _resolve_label_reference_workbook_path(cfg)
    if not workbook_path.exists():
        raise FileNotFoundError(
            "Label harmonization is enabled, but masterlist workbook was not found at "
            f"'{workbook_path}'."
        )

    masterlist_df = pd.read_excel(
        workbook_path,
        sheet_name=cfg.pretrain_masterlist_sheet_name,
        dtype=str,
    )
    if "ICD10h" not in masterlist_df.columns:
        raise KeyError(
            "Masterlist sheet must contain an 'ICD10h' column for label harmonization."
        )

    master_codes = {
        _normalize_icd10h_code_shape(code)
        for code in masterlist_df["ICD10h"].tolist()
        if _normalize_icd10h_code_shape(code)
    }

    try:
        transfer_df = pd.read_excel(
            workbook_path,
            sheet_name=cfg.pretrain_transfer_sheet_name,
            dtype=str,
        )
    except ValueError:
        transfer_df = pd.DataFrame(columns=["ICD10h_oct2020", "ICD10h2024"])

    if transfer_df.empty:
        return master_codes, {}

    required_columns = {"ICD10h_oct2020", "ICD10h2024"}
    missing_columns = required_columns.difference(transfer_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            "Transfer sheet is missing required columns for label harmonization: "
            f"{missing}."
        )

    transfer_pairs = transfer_df[["ICD10h_oct2020", "ICD10h2024"]].copy()
    transfer_pairs["ICD10h_oct2020"] = transfer_pairs["ICD10h_oct2020"].apply(
        _normalize_icd10h_code_shape
    )
    transfer_pairs["ICD10h2024"] = transfer_pairs["ICD10h2024"].apply(
        _normalize_icd10h_code_shape
    )
    transfer_pairs = transfer_pairs[
        (transfer_pairs["ICD10h_oct2020"] != "") & (transfer_pairs["ICD10h2024"] != "")
    ]

    transfer_map: dict[str, str] = {}
    for old_code, grouped in transfer_pairs.groupby("ICD10h_oct2020"):
        targets = sorted(set(grouped["ICD10h2024"].tolist()))
        if len(targets) > 1:
            rendered_targets = ", ".join(targets)
            raise ValueError(
                "Transfer sheet has ambiguous 2020->2024 mapping for "
                f"'{old_code}': {rendered_targets}."
            )
        transfer_map[old_code] = targets[0]

    return master_codes, transfer_map


def _map_code_with_transfer(code: str, transfer_map: Mapping[str, str]) -> str:
    """Map one label using transfer table and normalize shape."""
    normalized = _normalize_icd10h_code_shape(code)
    if not normalized:
        return ""
    mapped = transfer_map.get(normalized, normalized)
    mapped = _normalize_icd10h_code_shape(mapped)
    remapped = transfer_map.get(mapped, mapped)
    return _normalize_icd10h_code_shape(remapped)


def _harmonize_processed_labels(cfg: Config, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Map labels through transfer, normalize ICD10h shape, then drop unknown labels."""
    if dataframe.empty:
        return dataframe.reset_index(drop=True)

    master_codes, transfer_map = _load_label_reference_tables(cfg)
    if not master_codes:
        raise ValueError(
            "Masterlist has no valid ICD10h codes for label harmonization."
        )

    harmonized_codes: list[list[str]] = []
    keep_mask: list[bool] = []
    for raw_codes in dataframe["y_codes"].tolist():
        row_codes = _coerce_row_codes(raw_codes, cfg.label_separator)
        mapped_codes = [
            _map_code_with_transfer(code, transfer_map) for code in row_codes
        ]
        should_keep = bool(mapped_codes) and all(
            code in master_codes for code in mapped_codes
        )
        keep_mask.append(should_keep)
        harmonized_codes.append(mapped_codes)

    harmonized = dataframe.copy()
    harmonized["y_codes"] = harmonized_codes
    harmonized[cfg.dataset_label_column] = harmonized["y_codes"].apply(
        lambda codes: _build_label(codes, separator=cfg.label_separator)
    )
    return harmonized.loc[keep_mask].reset_index(drop=True)
