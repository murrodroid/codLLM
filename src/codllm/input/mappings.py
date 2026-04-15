from dataclasses import dataclass, field
from typing import Any

from codllm.data_augmentation import (
    accent_random_vowel,
    delete_random_char,
    insert_random_whitespace,
    qwerty_misspell,
    swap_adjacent_chars,
)


@dataclass
class DatasetMapping:
    """Describe how to map one source dataset into the canonical training schema."""

    text_col: int
    single_code_col: int
    multi_code_cols: list[int] = field(default_factory=list)
    sex_col: int | None = None
    sex_map: dict[str, str] = field(default_factory=dict)
    age_col: int | None = None
    record_id_col: int | None = None
    skip_rows: list[int] = field(default_factory=list)


BELGIUM_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=11,
    multi_code_cols=[12, 13, 14, 15, 16],
    sex_col=2,
    sex_map={"1": "male", "2": "female"},
    age_col=3,
    record_id_col=0,
    skip_rows=[0],
)

AMSTERDAM_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=7,
    multi_code_cols=[7, 9, 11, 13, 15, 17],
    sex_col=2,
    sex_map={
        "man": "male",
        "vrouw": "female",
        "m": "male",
        "v": "female",
        "1": "male",
        "2": "female",
    },
    age_col=3,
    record_id_col=0,
)

COPENHAGEN_MAPPING = DatasetMapping(
    text_col=37,
    single_code_col=39,
    multi_code_cols=[],
    sex_col=23,
    sex_map={"Mand": "male", "Kvinde": "female"},
    age_col=12,
    record_id_col=0,
    skip_rows=[0],
)

MASTERLIST_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=1,
    multi_code_cols=[],
    sex_col=None,
    sex_map={},
    age_col=None,
)

IPSWICH_MAPPING = DatasetMapping(
    text_col=4,
    single_code_col=5,
    multi_code_cols=[5, 6, 7, 8, 9, 10],
    sex_col=2,
    sex_map={"M": "male", "F": "female", "m": "male", "f": "female"},
    age_col=1,
    record_id_col=0,
)

MADRID_MAPPING = DatasetMapping(
    text_col=1,
    single_code_col=2,
    multi_code_cols=[2, 3],
    sex_col=4,
    sex_map={"1": "male", "2": "female"},
    age_col=5,
)

HISTORIC_STRINGS_MAPPING = DatasetMapping(
    text_col=1,
    single_code_col=2,
    multi_code_cols=[],
    record_id_col=0,
)

MAPPING_REGISTRY: dict[str, DatasetMapping] = {
    "belgium": BELGIUM_MAPPING,
    "amsterdam": AMSTERDAM_MAPPING,
    "copenhagen": COPENHAGEN_MAPPING,
    "masterlist": MASTERLIST_MAPPING,
    "ipswich": IPSWICH_MAPPING,
    "madrid": MADRID_MAPPING,
    "historic_strings": HISTORIC_STRINGS_MAPPING,
}

PERTURBATION_REGISTRY: dict[str, Any] = {
    "swap_adjacent_chars": swap_adjacent_chars,
    "delete_random_char": delete_random_char,
    "insert_random_whitespace": insert_random_whitespace,
    "accent_random_vowel": accent_random_vowel,
    "qwerty_misspell": qwerty_misspell,
}
