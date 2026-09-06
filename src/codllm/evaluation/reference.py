"""Training-only exposure inventories for reproducible evaluation slices."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from codllm.config import Config
from codllm.evaluation.provenance import cod_from_input, content_digest, normalize_cod


@dataclass
class EvaluationReference:
    """Compact original-training, augmentation, and masterlist exposure inventory."""

    historical_cods: set[str]
    adapted_cods: set[str]
    masterlist_cods: set[str]
    label_sources: dict[str, set[str]]
    label_languages: dict[str, set[str]]
    adaptation_languages: dict[str, set[str]]
    label_counts: dict[str, int]
    combinations: set[str]
    masterlist_labels: set[str]
    fingerprint: str = ""

    def payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable inventory without original text."""

        def convert(value: Any) -> Any:
            """Convert sets recursively into sorted JSON lists."""
            if isinstance(value, set):
                return sorted(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))


def cod_hash(value: str) -> str:
    """Hash a normalized description for comparison without exporting archival text."""
    return content_digest(normalize_cod(value))


def build_reference(
    original: pd.DataFrame,
    adapted: pd.DataFrame,
    masterlist: pd.DataFrame | None,
    cfg: Config,
) -> EvaluationReference:
    """Build exposure maps solely from the actual training partitions and adaptation data."""
    sources: dict[str, set[str]] = defaultdict(set)
    languages: dict[str, set[str]] = defaultdict(set)
    adaptation: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    historical_cods: set[str] = set()
    combinations: set[str] = set()
    historical = original.loc[
        original.get("data_role", pd.Series("historical", index=original.index)).eq(
            "historical"
        )
    ]
    for label, cod, source, language in zip(
        historical[cfg.dataset_label_column],
        historical["cod_key"],
        historical["source_id"],
        historical.get("language_candidates", pd.Series("und", index=historical.index)),
        strict=True,
    ):
        if not str(label).strip():
            continue
        codes = {
            part.strip()
            for part in str(label).split(cfg.label_separator)
            if part.strip()
        }
        historical_cods.add(cod_hash(cod))
        combinations.add(",".join(sorted(codes)))
        for code in codes:
            sources[code].add(str(source))
            languages[code].update(str(language).split(","))
            counts[code] += 1
    masterlist_labels: set[str] = set()
    for frame, is_masterlist in ((adapted, False), (masterlist, True)):
        if frame is None:
            continue
        for label, language in zip(
            frame[cfg.dataset_label_column].fillna("").astype(str),
            frame.get("language_candidates", pd.Series("und", index=frame.index))
            .fillna("und")
            .astype(str),
            strict=True,
        ):
            for code in {
                part.strip()
                for part in label.split(cfg.label_separator)
                if part.strip()
            }:
                adaptation[code].update(language.split(","))
                if is_masterlist:
                    masterlist_labels.add(code)
    adapted_cods = {
        cod_hash(cod_from_input(value, cfg))
        for value in adapted[cfg.dataset_text_column]
    }
    masterlist_cods = (
        {
            cod_hash(cod_from_input(value, cfg))
            for value in masterlist[cfg.dataset_text_column]
        }
        if masterlist is not None
        else set()
    )
    reference = EvaluationReference(
        historical_cods,
        adapted_cods,
        masterlist_cods,
        dict(sources),
        dict(languages),
        dict(adaptation),
        dict(counts),
        combinations,
        masterlist_labels,
    )
    reference.fingerprint = content_digest(reference.payload())
    return reference
