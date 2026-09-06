"""Private, content-addressed evaluation artifacts independent of W&B synchronization."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from codllm.config import Config
from codllm.evaluation.context import evaluation_rows
from codllm.evaluation.provenance import content_digest, provenance_signature
from codllm.evaluation.reference import EvaluationReference
from codllm.evaluation.scoring import score_publication


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a private JSON artifact with strict finite-number encoding."""

    def clean(value: Any) -> Any:
        """Represent unavailable numeric measurements as JSON null, not nonstandard NaN."""
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(clean(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write a compressed private dataframe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(directory: Path, name: str) -> pd.DataFrame:
    """Load a frozen partition and reject altered contents before evaluation."""
    contract = json.loads((directory / "contract.json").read_text())
    manifest = contract["manifests"].get(name)
    if manifest is None:
        raise ValueError(f"Frozen training contract has no '{name}' partition.")
    frame = pd.read_parquet(
        directory / "manifests" / f"{name}-{manifest['digest'][:16]}.parquet"
    )
    if (
        len(frame) != manifest["rows"]
        or content_digest(frame.fillna("").astype(str).to_dict("list"))
        != manifest["digest"]
    ):
        raise ValueError(f"Frozen '{name}' manifest was modified.")
    return frame


def persist_training_contract(
    cfg: Config, splits: Any, reference: EvaluationReference
) -> Path:
    """Freeze original partitions and training exposure; reject incompatible resume attempts."""
    directory = Path(cfg.output_dir) / "publication"
    payload = {
        "version": 1,
        "reference": reference.payload(),
        "languages": provenance_signature(cfg),
    }
    contract_path = directory / "reference.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != payload:
            raise ValueError(
                "Publication training exposure changed on resume; use a new output root."
            )
    else:
        write_json(contract_path, payload)
    manifests = {}
    for name in ("original_train", "val", "test", "holdout"):
        frame = getattr(splits, name, None)
        if frame is None:
            continue
        columns = [
            column
            for column in (
                "row_uid",
                cfg.dataset_text_column,
                "cod_text",
                "source_id",
                "record_id",
                "cod_key",
                "language",
                "language_candidates",
                cfg.dataset_label_column,
            )
            if column in frame
        ]
        manifest = frame[columns].fillna("").astype(str)
        digest = content_digest(manifest.to_dict("list"))
        manifests[name] = {"rows": len(frame), "digest": digest}
        path = directory / "manifests" / f"{name}-{digest[:16]}.parquet"
        if not path.exists():
            write_parquet(path, manifest)
    configuration = asdict(cfg)
    configuration.pop("hf_token", None)
    configuration["device"] = str(configuration["device"])
    configuration["device_map"] = str(configuration["device_map"])
    contract = {
        "version": 1,
        "protocol": cfg.evaluation_protocol,
        "manifests": manifests,
        "reference_fingerprint": reference.fingerprint,
    }
    recipe = {
        key: value
        for key, value in configuration.items()
        if key
        not in {
            "output_dir",
            "wandb",
            "device",
            "device_map",
            "max_runtime_seconds",
            "runtime_safety_margin_seconds",
            "auto_resume",
            "publication_gate",
            "publication_audit_top_groups",
            "publication_decisions_path",
            "evaluation_checkpoint",
            "evaluation_reference_dir",
            "evaluation_data_path",
            "evaluation_scope",
        }
    }
    contract["training_recipe_digest"] = content_digest(recipe)
    previous = directory / "contract.json"
    if previous.exists() and json.loads(previous.read_text()) != contract:
        raise ValueError(
            "Publication split manifests changed on resume; use a new output root."
        )
    write_json(previous, contract)
    write_json(directory / "effective_config.json", configuration)
    return directory


class PublicationReporter:
    """Compute versioned metrics during training and export only final selected-model predictions."""

    def __init__(self, cfg: Config, reference: EvaluationReference) -> None:
        self.cfg = cfg
        self.reference = reference
        self.exporting = False
        self.trainer: Any = None

    def __call__(
        self, predictions: list[set[str]], labels: list[set[str]]
    ) -> dict[str, float]:
        """Score one aligned evaluation and optionally retain its content-addressed artifacts."""
        from codllm.metrics import current_metric_artifact_scope

        rows = evaluation_rows.get()
        if rows is None:
            raise ValueError("Publication evaluation lost its original-row metadata.")
        metrics, records = score_publication(predictions, labels, rows, self.reference)
        if self.exporting and self.cfg.prediction_export_enabled:
            scope = current_metric_artifact_scope() or "evaluation"
            if any(
                part in {"..", "."} for part in scope.split("/")
            ) or scope.startswith("/"):
                raise ValueError("Invalid evaluation scope.")
            state = getattr(self.trainer, "state", None)
            identity = {
                "reference_fingerprint": self.reference.fingerprint,
                "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
                "best_metric": getattr(state, "best_metric", None),
                "global_step": getattr(state, "global_step", None),
                "epoch": getattr(state, "epoch", None),
                "scope": scope,
                "wandb_run_id": os.getenv("WANDB_RUN_ID"),
            }
            digest = content_digest({"identity": identity, "records": records})
            directory = (
                Path(self.cfg.output_dir) / "publication" / "predictions" / scope
            )
            path = directory / f"{digest}.parquet"
            if not path.exists():
                write_parquet(path, pd.DataFrame(records))
            payload = {
                "version": 1,
                "identity": identity,
                "metrics": metrics,
                "predictions": path.name,
                "digest": digest,
            }
            write_json(directory / f"{digest}.json", payload)
            write_json(directory / "selected.json", payload)
        return metrics
