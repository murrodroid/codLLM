"""Explicit publication phase gates and score-blind preparation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codllm.config import Config
from codllm.evaluation.artifacts import write_json
from codllm.evaluation.provenance import (
    annotate_provenance,
    content_digest,
    provenance_signature,
)
from codllm.experiments.specs import load_experiment_spec


def candidate_fingerprint(root: Path) -> str:
    """Hash the fully inherited paired candidate recipes, including language curation."""
    payload = {
        name: [
            run.env
            for run in load_experiment_spec(
                root / f"candidate_{name}.toml"
            ).expanded_runs()
        ]
        for name in ("winner", "runner_up")
    }
    defaults = Config()
    payload["languages"] = {
        name: provenance_signature(
            Config(
                evaluation_language_metadata_path=runs[0].get(
                    "CODLLM_EVALUATION_LANGUAGE_METADATA_PATH",
                    defaults.evaluation_language_metadata_path,
                ),
                evaluation_language_overrides_path=runs[0].get(
                    "CODLLM_EVALUATION_LANGUAGE_OVERRIDES_PATH"
                )
                or None,
            )
        )
        for name, runs in payload.items()
    }
    reduced = root / "reduced_protocol.toml"
    if reduced.exists():
        payload["reduced_protocol"] = reduced.read_text()
    return content_digest(payload)


def approve_stage(stage: str, path: str, note: str) -> dict[str, Any]:
    """Record an explicit reviewed phase decision without claiming experimental success."""
    if stage not in {"source_pair", "recipe", "reduced", "final"}:
        raise ValueError("Stage must be source_pair, recipe, reduced, or final.")
    if not note.strip():
        raise ValueError(
            "Record the evidence or decision in --note before approving a phase."
        )
    destination = Path(path)
    payload = (
        json.loads(destination.read_text())
        if destination.exists()
        else {"version": 1, "stages": {}}
    )
    payload["stages"][stage] = {
        "candidate_fingerprint": candidate_fingerprint(destination.parent),
        "note": note.strip(),
    }
    write_json(destination, payload)
    return payload["stages"][stage]


def validate_publication_config(cfg: Config) -> None:
    """Reject leakage-prone settings or unreviewed later-stage recipes before allocating work."""
    if not cfg.publication_eval_enabled:
        return
    if cfg.hold_out_dataset and cfg.hold_out_evaluate_per is not None:
        raise ValueError("Publication source folds require hold_out_evaluate_per=none.")
    if "historic_strings_en_2024" not in cfg.train_excluded_source_ids:
        raise ValueError(
            "Primary publication runs must exclude historic_strings_en_2024."
        )
    if cfg.hold_out_dataset in cfg.train_excluded_source_ids:
        raise ValueError(
            "The held-out archive must not also be listed as an excluded reference."
        )
    if not cfg.prediction_export_enabled:
        raise ValueError("Publication runs require selected-model prediction exports.")
    if cfg.publication_gate:
        path = Path(cfg.publication_decisions_path)
        if not path.exists():
            raise ValueError(
                f"Phase '{cfg.publication_gate}' is not reviewed; use invoke publication.approve first."
            )
        decision = (
            json.loads(path.read_text()).get("stages", {}).get(cfg.publication_gate)
        )
        if not decision or decision.get(
            "candidate_fingerprint"
        ) != candidate_fingerprint(path.parent):
            raise ValueError(
                f"Phase '{cfg.publication_gate}' approval is missing or stale after candidate edits."
            )
    if cfg.final_test_eval_enabled and cfg.publication_gate != "final":
        raise ValueError(
            "Final test metrics require a reviewed final publication gate."
        )
    provenance_signature(cfg)


def audit_dataset(cfg: Config, output: str) -> dict[str, Any]:
    """Write a score-blind source/language/support inventory from processed original data."""
    from codllm.data import DataHandler

    handler = DataHandler(cfg)
    frame = annotate_provenance(handler.ensure_processed(), cfg)
    rows = []
    for source, group in frame.groupby("source_id", sort=True):
        labels = (
            group[cfg.dataset_label_column].str.split(cfg.label_separator).explode()
        )
        rows.append(
            {
                "source_id": source,
                "rows": len(group),
                "unique_cods": group["cod_key"].nunique(),
                "missing_cod": int(group["cod_key"].eq("").sum()),
                "codes": labels.nunique(),
                "languages": group["language"].value_counts().to_dict(),
                "natural_multicod_rows": int(
                    group[cfg.dataset_label_column]
                    .str.contains(cfg.label_separator, regex=False)
                    .sum()
                ),
                "codes_with_fewer_than_10_rows": int(
                    (labels.value_counts() < 10).sum()
                ),
            }
        )
    overlap = {}
    sets = {
        source: set(group["cod_key"]) - {""}
        for source, group in frame.groupby("source_id")
    }
    for first in sorted(sets):
        for second in sorted(sets):
            if first < second:
                overlap[f"{first}__{second}"] = len(sets[first] & sets[second])
    payload = {
        "version": 1,
        "sources": rows,
        "cod_overlap": overlap,
        "language_signatures": provenance_signature(cfg),
    }
    write_json(Path(output), payload)
    return payload
