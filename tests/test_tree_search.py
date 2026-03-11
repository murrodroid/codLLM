"""Tests for experiments.tree_search.tree_search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from experiments.tree_search.icd10h_tree import NodeLevel, TreeNode
from experiments.tree_search.tree_search import (
    BeamItem,
    LLMChoice,
    LLMClient,
    SearchResult,
    TreeSearchConfig,
    beam_search,
    evaluate_search_results,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_tree() -> TreeNode:
    """3-branch tree: root -> 2 chapters -> auto-expand to leaves.

    Structure:
        ROOT
        ├── A (Infectious)
        │   └── A0 (Cholera group)        ← single child, auto-expand
        │       └── A00 (Cholera)          ← single child, auto-expand
        │           └── A00.0 (Cholera cl) ← single child, auto-expand
        │               └── A00.000 (leaf)
        └── B (Other infectious)
            └── B0 (Varicella group)       ← single child, auto-expand
                └── B01 (Varicella)        ← single child, auto-expand
                    └── B01.0 (Varicella)  ← single child, auto-expand
                        └── B01.000 (leaf)
    """
    root = TreeNode(code="ROOT", label="Root", level=NodeLevel.ROOT)

    # Branch A — each intermediate level has exactly 1 child (auto-expand)
    ch_a = TreeNode(code="A", label="Infectious", level=NodeLevel.CHAPTER, parent=root)
    bg_a0 = TreeNode(code="A0", label="Cholera group", level=NodeLevel.BLOCK_GROUP, parent=ch_a)
    blk_a00 = TreeNode(code="A00", label="Cholera", level=NodeLevel.BLOCK, parent=bg_a0)
    cat_a000 = TreeNode(code="A00.0", label="Cholera classical", level=NodeLevel.CATEGORY, parent=blk_a00)
    leaf_a = TreeNode(code="A00.000", label="Cholera biovar cholerae", level=NodeLevel.LEAF, parent=cat_a000)
    cat_a000.children = [leaf_a]
    blk_a00.children = [cat_a000]
    bg_a0.children = [blk_a00]
    ch_a.children = [bg_a0]

    # Branch B
    ch_b = TreeNode(code="B", label="Other infectious", level=NodeLevel.CHAPTER, parent=root)
    bg_b0 = TreeNode(code="B0", label="Varicella group", level=NodeLevel.BLOCK_GROUP, parent=ch_b)
    blk_b01 = TreeNode(code="B01", label="Varicella", level=NodeLevel.BLOCK, parent=bg_b0)
    cat_b010 = TreeNode(code="B01.0", label="Varicella encephalitis", level=NodeLevel.CATEGORY, parent=blk_b01)
    leaf_b = TreeNode(code="B01.000", label="Varicella encephalitis", level=NodeLevel.LEAF, parent=cat_b010)
    cat_b010.children = [leaf_b]
    blk_b01.children = [cat_b010]
    bg_b0.children = [blk_b01]
    ch_b.children = [bg_b0]

    root.children = [ch_a, ch_b]
    return root


def _mock_llm_always_first() -> LLMClient:
    """Return a mock LLM client that always picks the first option."""
    config = TreeSearchConfig()
    client = LLMClient(config)
    client.pick_children = MagicMock(  # type: ignore[assignment]
        side_effect=lambda cod, options, top_n: [
            LLMChoice(code=code) for code, _ in options[:top_n]
        ]
    )
    return client


def _mock_llm_picks(codes: list[str]) -> LLMClient:
    """Return a mock LLM client that picks specific codes each call."""
    config = TreeSearchConfig()
    client = LLMClient(config)
    call_idx = {"i": 0}

    def pick(cod: str, options: list[tuple[str, str]], top_n: int) -> list[LLMChoice]:
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(codes):
            return [LLMChoice(code=codes[idx])]
        return [LLMChoice(code=options[0][0])]

    client.pick_children = MagicMock(side_effect=pick)  # type: ignore[assignment]
    return client


# ---------------------------------------------------------------------------
# Beam search tests
# ---------------------------------------------------------------------------


class TestBeamSearch:
    def test_returns_leaf_codes(self) -> None:
        tree = _simple_tree()
        llm = _mock_llm_always_first()
        config = TreeSearchConfig(beam_width=1, top_choices=1)
        result = beam_search("cholera", tree, llm, config)
        assert len(result.predictions) == 1
        assert result.predictions[0] == "A00.000"

    def test_auto_expands_single_child(self) -> None:
        """Only the root→chapter step needs an LLM call; rest auto-expand."""
        tree = _simple_tree()
        llm = _mock_llm_always_first()
        config = TreeSearchConfig(beam_width=1, top_choices=1)
        result = beam_search("cholera", tree, llm, config)
        # Only 1 LLM call at root (2 chapters to choose from)
        assert result.llm_calls == 1

    def test_traces_record_path(self) -> None:
        tree = _simple_tree()
        llm = _mock_llm_always_first()
        config = TreeSearchConfig(beam_width=1, top_choices=1)
        result = beam_search("cholera", tree, llm, config)
        assert len(result.traces) == 1
        codes_in_trace = [code for code, _ in result.traces[0]]
        assert "A" in codes_in_trace
        assert "A00.000" in codes_in_trace

    def test_beam_width_limits_results(self) -> None:
        tree = _simple_tree()
        llm = _mock_llm_always_first()
        config = TreeSearchConfig(beam_width=2, top_choices=2)
        result = beam_search("disease", tree, llm, config)
        assert len(result.predictions) <= 2

    def test_picks_second_branch(self) -> None:
        tree = _simple_tree()
        llm = _mock_llm_picks(["B"])  # pick chapter B at root
        config = TreeSearchConfig(beam_width=1, top_choices=1)
        result = beam_search("varicella", tree, llm, config)
        assert result.predictions == ["B01.000"]


# ---------------------------------------------------------------------------
# LLM client parsing tests
# ---------------------------------------------------------------------------


class TestLLMClientParsing:
    def test_parse_valid_json(self) -> None:
        config = TreeSearchConfig()
        client = LLMClient(config)
        raw = '{"choices": [{"code": "A", "reason": "infectious"}]}'
        choices = client._parse_response(raw, {"A", "B"})
        assert len(choices) == 1
        assert choices[0].code == "A"

    def test_parse_filters_invalid_codes(self) -> None:
        config = TreeSearchConfig()
        client = LLMClient(config)
        raw = '{"choices": [{"code": "Z", "reason": "wrong"}, {"code": "A", "reason": "ok"}]}'
        choices = client._parse_response(raw, {"A", "B"})
        assert len(choices) == 1
        assert choices[0].code == "A"

    def test_parse_strips_markdown_fences(self) -> None:
        config = TreeSearchConfig()
        client = LLMClient(config)
        raw = '```json\n{"choices": [{"code": "B", "reason": "test"}]}\n```'
        choices = client._parse_response(raw, {"A", "B"})
        assert len(choices) == 1
        assert choices[0].code == "B"

    def test_parse_returns_empty_on_garbage(self) -> None:
        config = TreeSearchConfig()
        client = LLMClient(config)
        choices = client._parse_response("not json at all", {"A"})
        assert choices == []

    def test_build_messages_has_system_and_user(self) -> None:
        config = TreeSearchConfig()
        client = LLMClient(config)
        msgs = client._build_messages("cholera", [("A", "Infectious")], top_n=1)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "cholera" in msgs[1]["content"]
        assert "[A]" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_hit_at_1(self) -> None:
        assert recall_at_k("A00.000", ["A00.000", "B01.000"], 1) == 1.0

    def test_miss_at_1(self) -> None:
        assert recall_at_k("A00.000", ["B01.000", "A00.000"], 1) == 0.0

    def test_hit_at_3(self) -> None:
        assert recall_at_k("A00.000", ["B01.000", "C02.000", "A00.000"], 3) == 1.0

    def test_miss_beyond_k(self) -> None:
        assert recall_at_k("A00.000", ["B01.000", "C02.000", "A00.000"], 2) == 0.0

    def test_empty_predictions(self) -> None:
        assert recall_at_k("A00.000", [], 5) == 0.0


class TestEvaluateSearchResults:
    def test_returns_dataframe(self) -> None:
        results = [
            SearchResult(cod_string="cholera", predictions=["A00.000"]),
            SearchResult(cod_string="varicella", predictions=["B01.000"]),
        ]
        gold = ["A00.000", "C99.000"]
        df = evaluate_search_results(results, gold, k_values=[1])
        assert len(df) == 1
        assert df.iloc[0]["K"] == 1
        assert df.iloc[0]["Recall_at_K"] == 0.5
        assert df.iloc[0]["N"] == 2
