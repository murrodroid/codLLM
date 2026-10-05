"""Check metadata-input partitions against frozen COD-only historical manifests."""

from pathlib import Path
from typing import Any

import pandas as pd

from codllm.config import Config
from codllm.data import DataHandler
from codllm.evaluation.artifacts import load_manifest, write_json
from codllm.evaluation.provenance import cod_from_input
from codllm.evaluation.split_audit import validate_shared_audit_configs
from codllm.evaluation.splitting import validate_group_integrity


def compare_partition(actual: pd.DataFrame, reference: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Compare ordered identities, original CODs, targets, and COD input segments only."""
    columns = ["row_uid", "source_id", "record_id", "cod_text", "cod_key", "language", "language_candidates"]
    checks = {
        "rows_equal": len(actual) == len(reference),
        "identity_unique": not actual["row_uid"].duplicated().any(),
    }
    for column in columns:
        checks[f"{column}_equal"] = actual[column].fillna("").astype(str).reset_index(drop=True).equals(
            reference[column].fillna("").astype(str).reset_index(drop=True)
        )

    def targets(value: Any) -> tuple[str, ...]:
        """Ignore only target ordering, which is shuffled after original partitioning."""
        return tuple(sorted(str(value).split(cfg.label_separator)))

    checks["targets_equal"] = actual[cfg.dataset_label_column].map(targets).reset_index(drop=True).equals(
        reference[cfg.dataset_label_column].map(targets).reset_index(drop=True)
    )
    checks["cod_input_equal"] = actual[cfg.dataset_text_column].map(
        lambda value: cod_from_input(value, cfg)
    ).reset_index(drop=True).equals(reference[cfg.dataset_text_column].reset_index(drop=True))
    return {"rows": len(actual), "reference_rows": len(reference), "passed": all(checks.values()), "checks": checks}


def audit_metadata_pairing(configs: list[Config], reference: Path, output: Path) -> dict[str, Any]:
    """Prepare one shared original partition and fail closed on frozen-baseline mismatches."""
    validate_shared_audit_configs(configs)
    cfg = configs[0]
    if cfg.training_input != ["cod", "age", "sex"] or cfg.evaluation_protocol != "cod":
        raise ValueError("Metadata pairing requires COD+age+sex input and COD-only grouping")
    handler = DataHandler(cfg)
    splits, _ = handler.prepare_original_splits(handler.ensure_processed())
    validate_group_integrity(splits, cfg)
    parts = {}
    for name in ("original_train", "val", "test"):
        parts[name] = compare_partition(getattr(splits, name), load_manifest(reference, name), cfg)
    report = {
        "version": 1,
        "reference": str(reference),
        "processed_path": str(handler.processed_path),
        "status": "matched" if all(part["passed"] for part in parts.values()) else "mismatch",
        "partitions": parts,
        "scope": "Original historical partitions only; no model, augmented-split build, or scientific approval.",
    }
    write_json(output, report)
    return report
