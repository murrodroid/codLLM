"""LLM-guided beam search over the ICD10h tree."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.tree_search.icd10h_tree import NodeLevel, TreeNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TreeSearchConfig:
    """Configuration for the LLM-guided tree search."""

    beam_width: int = 3
    top_choices: int = 3
    max_children_shown: int = 25
    retry_count: int = 1
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    llm_api_key: str = "lm-studio"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LLMChoice:
    """A single child chosen by the LLM at one tree level."""

    code: str
    reason: str = ""


@dataclass
class BeamItem:
    """One active beam: a path through the tree so far."""

    node: TreeNode
    path: list[tuple[str, str]] = field(default_factory=list)
    depth: int = 0


@dataclass
class SearchResult:
    """Result of tree search for a single CoD string."""

    cod_string: str
    predictions: list[str] = field(default_factory=list)
    traces: list[list[tuple[str, str]]] = field(default_factory=list)
    llm_calls: int = 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert medical coding assistant specializing in ICD-10 classification \
of historical causes of death.

Given a cause-of-death string and a list of candidate categories, select the \
{top_n} most likely matches. The cause-of-death string may be in Dutch, French, \
Latin, or other historical languages — map meaning, not spelling.

Respond with ONLY valid JSON in this exact format:
{{
  "choices": [
    {{"code": "<code>", "reason": "<brief reason>"}},
    ...
  ]
}}

Select exactly {top_n} choices ordered from most to least likely. \
Use ONLY codes from the provided options list.\
"""

USER_PROMPT = """\
Cause of death: "{cod_string}"

Select the {top_n} best matching categories:
{options_block}

JSON only.\
"""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """OpenAI-compatible client for local LLM inference (LM Studio / Ollama)."""

    def __init__(self, config: TreeSearchConfig) -> None:
        self.config = config
        from openai import OpenAI

        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.total_calls: int = 0
        self.total_tokens: int = 0

    def pick_children(
        self,
        cod_string: str,
        options: list[tuple[str, str]],
        top_n: int,
    ) -> list[LLMChoice]:
        """Ask the LLM to pick the *top_n* most relevant children.

        Retries up to ``config.retry_count`` on JSON parse failure.
        """
        valid_codes = {code for code, _ in options}
        messages = self._build_messages(cod_string, options, top_n)

        for attempt in range(1 + self.config.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
                self.total_calls += 1
                if response.usage:
                    self.total_tokens += response.usage.total_tokens

                raw = response.choices[0].message.content or ""
                choices = self._parse_response(raw, valid_codes)
                if choices:
                    return choices[:top_n]

            except Exception:
                logger.warning(
                    "LLM call failed (attempt %d/%d)",
                    attempt + 1,
                    1 + self.config.retry_count,
                    exc_info=True,
                )

        # Fallback: return first top_n options in order
        logger.warning("All LLM attempts failed, falling back to first %d options", top_n)
        return [LLMChoice(code=code) for code, _ in options[:top_n]]

    def _build_messages(
        self,
        cod_string: str,
        options: list[tuple[str, str]],
        top_n: int,
    ) -> list[dict[str, str]]:
        options_block = "\n".join(
            f"  [{code}] {label}" for code, label in options
        )
        system = SYSTEM_PROMPT.format(top_n=top_n)
        user = USER_PROMPT.format(
            cod_string=cod_string,
            top_n=top_n,
            options_block=options_block,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _parse_response(
        self, raw_text: str, valid_codes: set[str]
    ) -> list[LLMChoice]:
        """Parse the JSON response, filtering to valid codes only."""
        # Strip markdown fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("JSON parse failed: %s", text[:200])
            return []

        raw_choices: list[dict[str, Any]] = []
        if isinstance(data, dict):
            raw_choices = data.get("choices", [])
        elif isinstance(data, list):
            raw_choices = data

        choices: list[LLMChoice] = []
        for item in raw_choices:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", "")).strip()
            if code in valid_codes:
                choices.append(
                    LLMChoice(code=code, reason=str(item.get("reason", "")))
                )
        return choices


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def beam_search(
    cod_string: str,
    tree: TreeNode,
    llm: LLMClient,
    config: TreeSearchConfig,
) -> SearchResult:
    """Run LLM-guided beam search from root to leaves."""
    B = config.beam_width
    C = config.top_choices
    llm_calls = 0

    beams = [BeamItem(node=tree, path=[], depth=0)]

    # Expand until all beams are at leaves (max depth = 5 levels)
    for _ in range(10):  # safety cap
        if all(b.node.is_leaf for b in beams):
            break

        next_beams: list[BeamItem] = []
        for beam in beams:
            if beam.node.is_leaf:
                next_beams.append(beam)
                continue

            children = beam.node.children
            if len(children) == 1:
                # Auto-expand single child — no LLM call needed
                child = children[0]
                next_beams.append(
                    BeamItem(
                        node=child,
                        path=beam.path + [(child.code, child.label)],
                        depth=beam.depth + 1,
                    )
                )
                continue

            options = beam.node.get_child_options()
            choices = llm.pick_children(cod_string, options, top_n=C)
            llm_calls += 1

            for choice in choices:
                child_node = _find_child(beam.node, choice.code)
                if child_node is None:
                    continue
                next_beams.append(
                    BeamItem(
                        node=child_node,
                        path=beam.path + [(child_node.code, child_node.label)],
                        depth=beam.depth + 1,
                    )
                )

        # Deduplicate by node code, keep first occurrence (LLM preference)
        seen: set[str] = set()
        unique: list[BeamItem] = []
        for b in next_beams:
            if b.node.code not in seen:
                seen.add(b.node.code)
                unique.append(b)
        beams = unique[:B]

    # Collect leaf predictions
    predictions: list[str] = []
    traces: list[list[tuple[str, str]]] = []
    for beam in beams:
        if beam.node.level == NodeLevel.LEAF:
            predictions.append(beam.node.code)
            traces.append(beam.path)

    return SearchResult(
        cod_string=cod_string,
        predictions=predictions,
        traces=traces,
        llm_calls=llm_calls,
    )


def _find_child(node: TreeNode, code: str) -> TreeNode | None:
    """Find a direct child of *node* matching *code*."""
    for child in node.children:
        if child.code == code:
            return child
    return None


def search_batch(
    cod_strings: list[str],
    tree: TreeNode,
    llm: LLMClient,
    config: TreeSearchConfig,
    progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 50,
) -> list[SearchResult]:
    """Run tree search for a batch of CoD strings.

    Parameters
    ----------
    checkpoint_path:
        If given, partial results are saved to this pickle file every
        *checkpoint_every* records **and** on KeyboardInterrupt.
        On restart, existing results are loaded and already-processed
        records are skipped.
    checkpoint_every:
        How often (in records) to write a checkpoint.
    """
    results: list[SearchResult] = []
    start_idx = 0

    # Resume from checkpoint if it exists
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            results = _load_checkpoint(checkpoint_path)
            start_idx = len(results)
            logger.info("Resumed from checkpoint: %d results loaded", start_idx)
            print(f"Resumed from checkpoint: {start_idx}/{len(cod_strings)} done")

    remaining = cod_strings[start_idx:]
    if not remaining:
        return results

    iterator: Any = remaining
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                remaining,
                desc="Tree search",
                initial=start_idx,
                total=len(cod_strings),
            )
        except ImportError:
            pass

    try:
        for i, cod in enumerate(iterator):
            results.append(beam_search(cod, tree, llm, config))

            if (
                checkpoint_path is not None
                and (i + 1) % checkpoint_every == 0
            ):
                _save_checkpoint(results, checkpoint_path)

    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(results)} / {len(cod_strings)} records.")
        if checkpoint_path is not None:
            _save_checkpoint(results, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        else:
            print("No checkpoint_path set — partial results returned in memory only.")

    # Final checkpoint
    if checkpoint_path is not None and results:
        _save_checkpoint(results, checkpoint_path)

    return results


def _save_checkpoint(results: list[SearchResult], path: Path) -> None:
    """Persist partial results to a pickle file."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    logger.debug("Checkpoint saved: %d results → %s", len(results), path)


def _load_checkpoint(path: Path) -> list[SearchResult]:
    """Load partial results from a pickle file."""
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def recall_at_k(gold_code: str, predictions: list[str], k: int) -> float:
    """Return 1.0 if *gold_code* appears in *predictions[:k]*, else 0.0."""
    return 1.0 if gold_code in predictions[:k] else 0.0


def evaluate_search_results(
    results: list[SearchResult],
    gold_codes: list[str],
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Recall@K for a batch of search results."""
    if k_values is None:
        k_values = [1, 3, 5]
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        recalls = [
            recall_at_k(gold, res.predictions, k)
            for gold, res in zip(gold_codes, results)
        ]
        rows.append(
            {"K": k, "Recall_at_K": sum(recalls) / max(len(recalls), 1), "N": len(recalls)}
        )
    return pd.DataFrame(rows)
