"""Fast unit tests for tokenization helpers and source_id plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from codllm.data.tokenization import (
    TokenizedSeq2SeqDataset,
    _tokenize_dataframe,
    _tokenize_dataframe_for_sequence_classification,
)


class _StubTokenizer:
    """Tokenizer stub: emits one int per character as fake input_ids."""

    def __call__(
        self,
        sources: list[str] | None = None,
        text_target: list[str] | None = None,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> dict[str, list[Any]]:
        target = text_target if text_target is not None else sources
        assert target is not None
        ids = [[i for i, _ in enumerate(text)] for text in target]
        attn = [[1] * len(seq) for seq in ids]
        return {"input_ids": ids, "attention_mask": attn}


@dataclass
class _StubConfig:
    """Minimal config exposing only the fields the tokenization helpers use."""

    max_source_length: int = 32
    dataset_text_column: str = "text"
    dataset_label_column: str = "label"


class TestTokenizedSeq2SeqDataset:
    def test_stores_source_ids_when_provided(self) -> None:
        ds = TokenizedSeq2SeqDataset(
            {"input_ids": [[1], [2]]},
            source_ids=["a", "b"],
        )
        assert ds.source_ids == ["a", "b"]

    def test_default_source_ids_is_none(self) -> None:
        ds = TokenizedSeq2SeqDataset({"input_ids": [[1], [2]]})
        assert ds.source_ids is None

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="source_ids length"):
            TokenizedSeq2SeqDataset(
                {"input_ids": [[1], [2]]}, source_ids=["only_one"]
            )

    def test_getitem_excludes_source_ids(self) -> None:
        """Per-row dicts must not contain string source_ids (collator-incompatible)."""
        ds = TokenizedSeq2SeqDataset(
            {"input_ids": [[1, 2], [3, 4]]},
            source_ids=["a", "b"],
        )
        item = ds[0]
        assert "source_id" not in item
        assert item == {"input_ids": [1, 2]}


class TestTokenizeDataframeSourceIds:
    def test_extracts_source_ids_from_dataframe(self) -> None:
        cfg = _StubConfig()
        tokenizer = _StubTokenizer()
        df = pd.DataFrame(
            {
                "text": ["alpha", "beta"],
                "label": ["A", "B"],
                "source_id": ["copenhagen", "amsterdam"],
            }
        )

        ds = _tokenize_dataframe(cfg, tokenizer, df, target_max_length=8)

        assert ds.source_ids == ["copenhagen", "amsterdam"]

    def test_works_when_source_id_column_missing(self) -> None:
        cfg = _StubConfig()
        tokenizer = _StubTokenizer()
        df = pd.DataFrame({"text": ["alpha"], "label": ["A"]})

        ds = _tokenize_dataframe(cfg, tokenizer, df, target_max_length=8)

        assert ds.source_ids is None

    def test_classification_path_extracts_source_ids(self) -> None:
        cfg = _StubConfig()
        tokenizer = _StubTokenizer()
        df = pd.DataFrame(
            {
                "text": ["alpha", "beta"],
                "label": ["A", "B"],
                "source_id": ["copenhagen", "amsterdam"],
            }
        )

        ds = _tokenize_dataframe_for_sequence_classification(
            cfg=cfg, tokenizer=tokenizer, dataframe=df, label2id={"A": 0, "B": 1}
        )

        assert ds.source_ids == ["copenhagen", "amsterdam"]
