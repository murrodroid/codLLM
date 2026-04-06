"""Tests for experiments.tree_search.icd10h_tree."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.tree_search.icd10h_tree import (
    NodeLevel,
    TreeNode,
    _summarise_labels,
    build_tree,
    get_node_by_code,
    tree_stats,
)

MASTERLIST_PATH = Path("data/raw/ICD10h_Masterlist_2024.xlsx")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mini_df() -> pd.DataFrame:
    """Minimal masterlist with 4 codes across 2 chapters."""
    return pd.DataFrame(
        {
            "ICD10h": ["A00.000", "A00.100", "A01.000", "B01.000"],
            "ICD10": ["A00.0", "A00.1", "A01.0", "B01.0"],
            "icd10_2levelCATEGORY": [
                "Cholera",
                "Cholera",
                "Typhoid and paratyphoid fevers",
                "Varicella [chickenpox]",
            ],
            "ICD10_2levelCAUSE": [
                "Cholera due to Vibrio cholerae 01, biovar cholerae",
                "Cholera due to Vibrio cholerae 01, biovar el tor",
                "Typhoid fever",
                "Varicella encephalitis",
            ],
            "ICD10h_DESCRIPTION": [
                "Cholera due to Vibrio cholerae 01, biovar cholerae",
                "Cholera due to Vibrio cholerae 01, biovar el tor",
                "Typhoid fever",
                "Varicella encephalitis",
            ],
            "HistCat": ["Diarrhoea", "Diarrhoea", "Diarrhoea", "Infectious"],
            "DoNotUse": [0, 0, 0, 0],
            "NotForUnderlying": [0, 0, 0, 0],
            "GenderSpecific": [0, 0, 0, 0],
        }
    )


def _mini_tree() -> TreeNode:
    """Build a tree from the mini fixture via a temp Excel file."""
    import tempfile

    df = _mini_df()
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df.to_excel(f.name, sheet_name="Masterlist", index=False)
        return build_tree(f.name)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestTreeNode:
    def test_leaf_has_no_children(self) -> None:
        node = TreeNode(code="A00.000", label="Cholera", level=NodeLevel.LEAF)
        assert node.is_leaf is True
        assert node.child_count == 0

    def test_parent_is_not_leaf(self) -> None:
        parent = TreeNode(code="A", label="Chapter A", level=NodeLevel.CHAPTER)
        child = TreeNode(
            code="A0", label="group", level=NodeLevel.BLOCK_GROUP, parent=parent
        )
        parent.children.append(child)
        assert parent.is_leaf is False
        assert parent.child_count == 1

    def test_get_child_options(self) -> None:
        parent = TreeNode(code="ROOT", label="root", level=NodeLevel.ROOT)
        c1 = TreeNode(
            code="A", label="Infectious", level=NodeLevel.CHAPTER, parent=parent
        )
        c2 = TreeNode(
            code="B", label="Other infectious", level=NodeLevel.CHAPTER, parent=parent
        )
        parent.children = [c1, c2]
        opts = parent.get_child_options()
        assert opts == [("A", "Infectious"), ("B", "Other infectious")]


class TestSummariseLabels:
    def test_short_list_no_ellipsis(self) -> None:
        assert _summarise_labels(["A", "B"]) == "A; B"

    def test_long_list_truncated(self) -> None:
        result = _summarise_labels(["A", "B", "C", "D", "E"], max_items=3)
        assert result == "A; B; C; ..."

    def test_exact_limit_no_ellipsis(self) -> None:
        result = _summarise_labels(["A", "B", "C"], max_items=3)
        assert result == "A; B; C"


class TestBuildTreeMini:
    def test_root_level(self) -> None:
        root = _mini_tree()
        assert root.level == NodeLevel.ROOT
        assert root.code == "ROOT"

    def test_two_chapters(self) -> None:
        root = _mini_tree()
        assert root.child_count == 2
        assert {c.code for c in root.children} == {"A", "B"}

    def test_all_levels_present(self) -> None:
        root = _mini_tree()
        levels_seen: set[NodeLevel] = set()

        def walk(n: TreeNode) -> None:
            levels_seen.add(n.level)
            for c in n.children:
                walk(c)

        walk(root)
        assert levels_seen == set(NodeLevel)

    def test_leaves_are_full_codes(self) -> None:
        root = _mini_tree()
        leaves: list[str] = []

        def walk(n: TreeNode) -> None:
            if n.is_leaf:
                leaves.append(n.code)
            for c in n.children:
                walk(c)

        walk(root)
        assert set(leaves) == {"A00.000", "A00.100", "A01.000", "B01.000"}

    def test_parent_references(self) -> None:
        root = _mini_tree()

        def walk(n: TreeNode) -> None:
            for c in n.children:
                assert c.parent is n
                walk(c)

        walk(root)


class TestGetNodeByCode:
    def test_find_root(self) -> None:
        root = _mini_tree()
        assert get_node_by_code(root, "ROOT") is root

    def test_find_leaf(self) -> None:
        root = _mini_tree()
        node = get_node_by_code(root, "A00.000")
        assert node is not None
        assert node.level == NodeLevel.LEAF

    def test_find_block_group(self) -> None:
        root = _mini_tree()
        node = get_node_by_code(root, "A0")
        assert node is not None
        assert node.level == NodeLevel.BLOCK_GROUP

    def test_missing_returns_none(self) -> None:
        root = _mini_tree()
        assert get_node_by_code(root, "Z99.999") is None


# ---------------------------------------------------------------------------
# Integration tests (real masterlist)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MASTERLIST_PATH.exists(), reason="Masterlist not present")
class TestBuildTreeIntegration:
    @pytest.fixture(scope="class")
    def root(self) -> TreeNode:
        return build_tree(MASTERLIST_PATH)

    def test_25_chapters(self, root: TreeNode) -> None:
        assert root.child_count == 25

    def test_block_groups_at_most_10(self, root: TreeNode) -> None:
        for chapter in root.children:
            assert chapter.child_count <= 10, f"Chapter {chapter.code}"

    def test_blocks_at_most_10(self, root: TreeNode) -> None:
        for ch in root.children:
            for bg in ch.children:
                assert bg.child_count <= 10, f"BlockGroup {bg.code}"

    def test_categories_at_most_10(self, root: TreeNode) -> None:
        for ch in root.children:
            for bg in ch.children:
                for blk in bg.children:
                    assert blk.child_count <= 10, f"Block {blk.code}"

    def test_leaves_at_most_18(self, root: TreeNode) -> None:
        for ch in root.children:
            for bg in ch.children:
                for blk in bg.children:
                    for cat in blk.children:
                        assert cat.child_count <= 18, f"Category {cat.code}"

    def test_total_leaves_14088(self, root: TreeNode) -> None:
        stats = tree_stats(root)
        assert stats["leaf"] == 14_088
