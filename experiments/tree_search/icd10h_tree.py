"""ICD10h hierarchy tree: data structures and builder from masterlist."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

ICD10H_PATTERN = re.compile(r"^[A-Z]\d{2}\.\d{3}$")

_MAX_LABEL_ITEMS = 4


class NodeLevel(Enum):
    """Hierarchy levels in the ICD10h tree."""

    ROOT = "root"
    CHAPTER = "chapter"
    BLOCK_GROUP = "block_group"
    BLOCK = "block"
    CATEGORY = "category"
    LEAF = "leaf"


@dataclass
class TreeNode:
    """A single node in the ICD10h hierarchy tree."""

    code: str
    label: str
    level: NodeLevel
    children: list[TreeNode] = field(default_factory=list, repr=False)
    parent: TreeNode | None = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def child_count(self) -> int:
        return len(self.children)

    def get_child_options(self) -> list[tuple[str, str]]:
        """Return (code, label) pairs for all children."""
        return [(c.code, c.label) for c in self.children]


def _summarise_labels(names: list[str], max_items: int = _MAX_LABEL_ITEMS) -> str:
    """Join up to *max_items* names with '; ', appending '...' if truncated."""
    if len(names) <= max_items:
        return "; ".join(names)
    return "; ".join(names[:max_items]) + "; ..."


def load_masterlist_df(path: str | Path) -> pd.DataFrame:
    """Load the masterlist and add derived hierarchy columns."""
    df = pd.read_excel(path, sheet_name="Masterlist", engine="openpyxl")
    mask = df["ICD10h"].astype(str).str.match(ICD10H_PATTERN)
    df = df[mask].copy()
    df["chapter"] = df["ICD10h"].str[0]
    df["block_group"] = df["ICD10h"].str[:2]
    df["block"] = df["ICD10h"].str[:3]
    df["category"] = df["ICD10h"].str[:5]
    df["leaf"] = df["ICD10h"]
    return df


def build_tree(masterlist_path: str | Path) -> TreeNode:
    """Build the full 5-level ICD10h tree from the Excel masterlist.

    Hierarchy: Root → Chapter → Block Group → Block → Category → Leaf.
    """
    df = load_masterlist_df(masterlist_path)
    root = TreeNode(code="ROOT", label="ICD10h Classification", level=NodeLevel.ROOT)

    for chapter_code in sorted(df["chapter"].unique()):
        chapter_df = df[df["chapter"] == chapter_code]
        chapter_cats = sorted(chapter_df["icd10_2levelCATEGORY"].dropna().unique())
        chapter_label = _summarise_labels(chapter_cats)
        chapter_node = TreeNode(
            code=chapter_code,
            label=chapter_label,
            level=NodeLevel.CHAPTER,
            parent=root,
        )
        root.children.append(chapter_node)

        for bg_code in sorted(chapter_df["block_group"].unique()):
            bg_df = chapter_df[chapter_df["block_group"] == bg_code]
            bg_cats = sorted(bg_df["icd10_2levelCATEGORY"].dropna().unique())
            bg_label = _summarise_labels(bg_cats)
            bg_node = TreeNode(
                code=bg_code,
                label=bg_label,
                level=NodeLevel.BLOCK_GROUP,
                parent=chapter_node,
            )
            chapter_node.children.append(bg_node)

            for block_code in sorted(bg_df["block"].unique()):
                block_df = bg_df[bg_df["block"] == block_code]
                block_label = str(block_df["icd10_2levelCATEGORY"].iloc[0])
                block_node = TreeNode(
                    code=block_code,
                    label=block_label,
                    level=NodeLevel.BLOCK,
                    parent=bg_node,
                )
                bg_node.children.append(block_node)

                for cat_code in sorted(block_df["category"].unique()):
                    cat_df = block_df[block_df["category"] == cat_code]
                    cat_label = str(cat_df["ICD10_2levelCAUSE"].iloc[0])
                    cat_node = TreeNode(
                        code=cat_code,
                        label=cat_label,
                        level=NodeLevel.CATEGORY,
                        parent=block_node,
                    )
                    block_node.children.append(cat_node)

                    for _, row in cat_df.iterrows():
                        leaf_node = TreeNode(
                            code=row["ICD10h"],
                            label=str(row["ICD10h_DESCRIPTION"]),
                            level=NodeLevel.LEAF,
                            parent=cat_node,
                        )
                        cat_node.children.append(leaf_node)

    return root


def get_node_by_code(root: TreeNode, code: str) -> TreeNode | None:
    """Look up a node by its code string via prefix-based traversal."""
    if root.code == code:
        return root
    for child in root.children:
        if code.startswith(child.code) or child.code == code:
            result = get_node_by_code(child, code)
            if result is not None:
                return result
    return None


def tree_stats(root: TreeNode) -> dict[str, int]:
    """Return summary statistics: node counts per level and max branching."""
    counts: dict[str, int] = {level.value: 0 for level in NodeLevel}
    max_branching = 0

    def _walk(node: TreeNode) -> None:
        nonlocal max_branching
        counts[node.level.value] += 1
        if node.child_count > max_branching:
            max_branching = node.child_count
        for child in node.children:
            _walk(child)

    _walk(root)
    counts["total_nodes"] = sum(counts.values())
    counts["max_branching_factor"] = max_branching
    return counts
