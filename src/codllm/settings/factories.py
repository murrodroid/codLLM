from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from codllm.settings.options import SUPPORTED_TRAINING_INPUTS
from codllm.settings.types import TrainingInput

if TYPE_CHECKING:
    from codllm.settings.schema import DataSourceConfig


def default_device() -> torch.device:
    """Choose the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_device_map() -> Optional[str]:
    """Choose a sensible default device map for the current hardware."""
    if torch.cuda.is_available():
        return "auto"
    return None


def default_data_sources() -> list[DataSourceConfig]:
    """Return default data sources expected in the raw data directory."""
    from codllm.settings.schema import DataSourceConfig

    return [
        DataSourceConfig(
            source_id="belgium_1920_1930",
            path="SOSA_EXTR_1920-1930 (belgium).xlsx",
            mapping_id="belgium",
        ),
        DataSourceConfig(
            source_id="amsterdam_1854_1926",
            path="AMC_1854_1926_LM.csv",
            mapping_id="amsterdam",
            sep=";",
        ),
        DataSourceConfig(
            source_id="copenhagen_may2025",
            path="Copenhagen_burials_all_May2025.csv",
            mapping_id="copenhagen",
        ),
        DataSourceConfig(
            source_id="ipswich_1871_1911",
            path="Ipswich_deaths_codllm.txt",
            mapping_id="ipswich",
            file_type="csv",
            sep="|",
            encoding="latin-1",
        ),
        DataSourceConfig(
            source_id="madrid_1905_1927",
            path="Madrid 1905_1927.csv",
            mapping_id="madrid",
        ),
        DataSourceConfig(
            source_id="historic_strings_en_2024",
            path="ICD10H_HISTORICSTRINGSENGLISH_2024.2.xlsx",
            mapping_id="historic_strings",
            sheet_name="HistoricstringsEnglish2024 1.1",
        ),
    ]


def default_training_input() -> list[TrainingInput]:
    """Return the default training input field order."""
    return list(SUPPORTED_TRAINING_INPUTS)


def default_input_field_prefixes() -> dict[TrainingInput, str]:
    """Return default text prefixes for processed training input fields."""
    return {
        "cod": "cod: ",
        "age": "age: ",
        "sex": "sex: ",
    }


def default_pretrain_perturbations() -> list[str]:
    """Return default perturbations for synthetic pretraining rows."""
    return [
        "swap_adjacent_chars",
        "delete_random_char",
        "accent_random_vowel",
        "qwerty_misspell",
    ]


def default_masterlist_inject_perturbations() -> list[str]:
    """Return default perturbations for masterlist injection rows."""
    return [
        "swap_adjacent_chars",
        "delete_random_char",
        "accent_random_vowel",
        "qwerty_misspell",
    ]


def default_balance_perturbations() -> list[str]:
    """Return default perturbations for balance-policy row manipulation."""
    return [
        "swap_adjacent_chars",
        "delete_random_char",
        "accent_random_vowel",
        "qwerty_misspell",
    ]
