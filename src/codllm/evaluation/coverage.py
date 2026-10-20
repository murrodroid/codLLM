"""Label coverage from checksum-verified frozen original publication partitions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from codllm.config import Config
from codllm.evaluation.artifacts import load_manifest
from codllm.evaluation.provenance import content_digest


def label_sets(frame: pd.DataFrame, cfg: Config) -> list[set[str]]:
    """Parse full ICD10h labels using the training configuration's column and separator."""
    return [
        {part.strip() for part in value.split(cfg.label_separator) if part.strip()}
        for value in frame[cfg.dataset_label_column].fillna("").astype(str)
    ]


def partition_identity(frame: pd.DataFrame, cfg: Config) -> str:
    """Fingerprint original row/COD/label identities independently of formatted metadata inputs."""
    columns = ["row_uid", "source_id", "record_id", "cod_key", cfg.dataset_label_column]
    return content_digest(frame.sort_values("row_uid")[columns].to_dict("list"))


def summarize_coverage(
    frame: pd.DataFrame,
    historical: set[str],
    source_labels: dict[str, set[str]],
    masterlist: set[str],
    adaptation: set[str],
    cfg: Config,
) -> dict[str, Any]:
    """Count distinct codes, target occurrences, and fully covered rows separately."""
    labels = label_sets(frame, cfg)
    codes = set().union(*labels) if labels else set()
    unseen = [target - historical for target in labels]
    target_count = sum(map(len, labels))
    unseen_targets = sum(map(len, unseen))
    all_seen_rows = sum(not values for values in unseen)
    return {
        "rows": len(frame),
        "distinct_codes": len(codes),
        "historically_seen_codes": len(codes & historical),
        "historically_unseen_codes": len(codes - historical),
        "distinct_code_coverage": len(codes & historical) / len(codes)
        if codes
        else None,
        "targets": target_count,
        "historically_seen_targets": target_count - unseen_targets,
        "historically_unseen_targets": unseen_targets,
        "target_coverage": 1 - unseen_targets / target_count if target_count else None,
        "rows_all_labels_seen": all_seen_rows,
        "rows_any_unseen_label": len(frame) - all_seen_rows,
        "rows_no_labels_seen": sum(
            bool(target) and not target & historical for target in labels
        ),
        "fully_covered_row_fraction": all_seen_rows / len(frame)
        if len(frame)
        else None,
        "historically_unseen_codes_in_masterlist": len(
            (codes - historical) & masterlist
        ),
        "codes_absent_from_all_adaptation": len(codes - adaptation),
        "targets_absent_from_all_adaptation": sum(
            len(target - adaptation) for target in labels
        ),
        "targets_seen_in_same_source": sum(
            len(target & source_labels.get(source, set()))
            for target, source in zip(labels, frame["source_id"], strict=True)
        ),
    }


def label_coverage(directory: Path, cfg: Config | None = None) -> dict[str, Any]:
    """Audit frozen train/validation/test labels without predictions or reconstructing splits."""
    cfg = cfg or Config()
    contract = json.loads((directory / "contract.json").read_text())
    reference = json.loads((directory / "reference.json").read_text())["reference"]
    fingerprint = reference["fingerprint"]
    if content_digest({**reference, "fingerprint": ""}) != fingerprint:
        raise ValueError("Frozen exposure reference fingerprint differs.")
    if fingerprint != contract["reference_fingerprint"]:
        raise ValueError("Exposure reference does not match the frozen contract.")
    train = load_manifest(directory, "original_train")
    train_labels = label_sets(train, cfg)
    historical_counts = Counter(code for target in train_labels for code in target)
    if dict(historical_counts) != reference["label_counts"]:
        raise ValueError(
            "Original training labels differ from the frozen exposure inventory."
        )
    historical = set(historical_counts)
    source_labels: dict[str, set[str]] = defaultdict(set)
    for target, source in zip(train_labels, train["source_id"], strict=True):
        source_labels[source].update(target)
    masterlist, adaptation = (
        set(reference["masterlist_labels"]),
        set(reference["adaptation_languages"]),
    )
    train_uids, train_cods = set(train["row_uid"]), set(train["cod_key"])
    partitions = {}
    for split in ("val", "test", "holdout"):
        if split not in contract["manifests"]:
            continue
        frame = load_manifest(directory, split)
        if set(frame["row_uid"]) & train_uids:
            raise ValueError(f"{split} shares row identities with training.")
        shared_cods = len(set(frame["cod_key"]) & train_cods)
        if contract["protocol"] == "cod" and split != "holdout" and shared_cods:
            raise ValueError(f"{split} shares normalized CODs with grouped training.")
        partitions[split] = {
            "manifest": contract["manifests"][split],
            "identity_digest": partition_identity(frame, cfg),
            "distinct_cods": int(frame["cod_key"].nunique()),
            "cods_shared_with_training": shared_cods,
            "all": summarize_coverage(
                frame, historical, source_labels, masterlist, adaptation, cfg
            ),
            "sources": {
                source: summarize_coverage(
                    part, historical, source_labels, masterlist, adaptation, cfg
                )
                for source, part in frame.groupby("source_id", sort=True)
            },
        }
    return {
        "publication_directory": str(directory),
        "protocol": contract["protocol"],
        "reference_fingerprint": fingerprint,
        "training_manifest": contract["manifests"]["original_train"],
        "training_identity_digest": partition_identity(train, cfg),
        "training_distinct_codes": len(historical),
        "training_distinct_cods": len(train_cods),
        "masterlist_distinct_codes": len(masterlist),
        "partitions": partitions,
        "scope": "Frozen original historical training labels; masterlist/adaptation inventories reported separately.",
    }
