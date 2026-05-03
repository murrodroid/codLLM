from pathlib import Path
from typing import NamedTuple

import pandas as pd

from codllm.runtime.paths import resolve_source_path
from codllm.settings.schema import Config
from codllm.input.transform import (
    ICD10H_CANONICAL_PATTERN,
    _build_label,
    _coerce_row_codes,
    _normalize_icd10h_code_shape,
)


class LabelHarmonizationReference(NamedTuple):
    """Reference tables required to standardize ICD10h labels."""

    master_codes: frozenset[str]
    transfer_map: dict[str, str]


def resolve_label_harmonization_workbook_path(cfg: Config) -> Path:
    """Resolve the ICD10h reference workbook used for label harmonization."""
    reference_path = Path(cfg.label_harmonization_masterlist_path)
    if reference_path.is_absolute():
        return reference_path

    data_root = Path(cfg.data_raw_dir)
    if reference_path.parts and data_root.parts:
        if reference_path.parts[0] == data_root.parts[-1]:
            trimmed_candidate = data_root / Path(*reference_path.parts[1:])
            if trimmed_candidate.exists():
                return trimmed_candidate

    return resolve_source_path(
        cfg.label_harmonization_masterlist_path, cfg.data_raw_dir
    )


def _load_label_reference_tables(cfg: Config) -> LabelHarmonizationReference:
    """Load masterlist-valid labels and 2020->2024 transfer mapping."""
    workbook_path = resolve_label_harmonization_workbook_path(cfg)
    if not workbook_path.exists():
        raise FileNotFoundError(
            "Label harmonization is enabled, but masterlist workbook was not found at "
            f"'{workbook_path}'."
        )

    masterlist_df = pd.read_excel(
        workbook_path,
        sheet_name=cfg.label_harmonization_masterlist_sheet_name,
        dtype=str,
    )
    if "ICD10h" not in masterlist_df.columns:
        raise KeyError(
            "Masterlist sheet must contain an 'ICD10h' column for label harmonization."
        )

    master_codes = frozenset(
        normalized
        for raw_code in masterlist_df["ICD10h"].tolist()
        if (normalized := _normalize_icd10h_code_shape(raw_code))
        and ICD10H_CANONICAL_PATTERN.fullmatch(normalized)
    )

    transfer_df = pd.read_excel(
        workbook_path,
        sheet_name=cfg.label_harmonization_transfer_sheet_name,
        dtype=str,
    )

    if transfer_df.empty:
        return LabelHarmonizationReference(master_codes=master_codes, transfer_map={})

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

    return LabelHarmonizationReference(
        master_codes=master_codes,
        transfer_map=transfer_map,
    )


def _map_code_with_transfer(code: str, transfer_map: dict[str, str]) -> str:
    """Map one label using transfer table and normalize shape."""
    normalized = _normalize_icd10h_code_shape(code)
    if not normalized:
        return ""
    mapped = transfer_map.get(normalized, normalized)
    return _normalize_icd10h_code_shape(mapped)


def _harmonize_row_codes(
    raw_codes: object,
    cfg: Config,
    reference: LabelHarmonizationReference,
) -> list[str]:
    """Return one harmonized row label list, or an empty list when any label is invalid."""
    harmonized_codes: list[str] = []
    seen_codes: set[str] = set()
    for raw_code in _coerce_row_codes(raw_codes, cfg.label_separator):
        mapped_code = _map_code_with_transfer(raw_code, reference.transfer_map)
        if mapped_code not in reference.master_codes:
            return []
        if mapped_code not in seen_codes:
            harmonized_codes.append(mapped_code)
            seen_codes.add(mapped_code)
    if len(harmonized_codes) > cfg.max_label_count:
        return []
    return harmonized_codes


def _harmonize_processed_labels(cfg: Config, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Map labels through transfer, normalize ICD10h shape, then drop unknown labels."""
    if dataframe.empty:
        return dataframe.reset_index(drop=True)

    if "y_codes" not in dataframe.columns:
        raise KeyError(
            "Processed dataframe must contain 'y_codes' for label harmonization."
        )

    reference = _load_label_reference_tables(cfg)
    if not reference.master_codes:
        raise ValueError(
            "Masterlist has no valid ICD10h codes for label harmonization."
        )

    harmonized_codes: list[list[str]] = []
    keep_mask: list[bool] = []
    for raw_codes in dataframe["y_codes"].tolist():
        mapped_codes = _harmonize_row_codes(raw_codes, cfg, reference)
        keep_mask.append(bool(mapped_codes))
        harmonized_codes.append(mapped_codes)

    harmonized = dataframe.copy()
    harmonized["y_codes"] = harmonized_codes
    harmonized[cfg.dataset_label_column] = harmonized["y_codes"].apply(
        lambda codes: _build_label(codes, separator=cfg.label_separator)
    )
    return harmonized.loc[keep_mask].reset_index(drop=True)
