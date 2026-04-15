from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Mapping

from codllm.config import Config
from codllm.data_handler import DataHandler, DataSplits
from codllm.training.stages import TrainingStage
from transformers import TrainingArguments


def dataset_row_count(dataset: Any) -> int | None:
    """Return dataset row count when available."""
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except (TypeError, ValueError):
        return None


def _source_distribution(dataset: Any) -> dict[str, int] | None:
    """Return per-source counts when the split contains source metadata."""
    if dataset is None or not hasattr(dataset, "columns"):
        return None
    if "source_id" not in dataset.columns:
        return None
    counts = dataset["source_id"].fillna("unknown").astype(str).value_counts()
    return {source: int(count) for source, count in counts.items()}


def _label_stats(dataset: Any, label_column: str) -> dict[str, int] | None:
    """Return non-empty and unique label counts for a split."""
    if dataset is None or not hasattr(dataset, "columns"):
        return None
    if label_column not in dataset.columns:
        return None
    labels = dataset[label_column].fillna("").astype(str).str.strip()
    non_empty = labels[labels != ""]
    return {
        "non_empty_count": int(non_empty.shape[0]),
        "unique_count": int(non_empty.nunique()),
    }


def _load_json_file(path: Path) -> dict[str, Any] | None:
    """Load JSON file contents when present and parseable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build_data_metadata(
    cfg: Config,
    splits: DataSplits,
    force_reprocess: bool,
    handler: DataHandler,
) -> dict[str, Any]:
    """Build split and processed-data metadata for one training run."""
    source_distribution = {
        "train": _source_distribution(splits.train),
        "val": _source_distribution(splits.val),
        "test": _source_distribution(splits.test),
    }
    label_stats = {
        "train": _label_stats(splits.train, cfg.dataset_label_column),
        "val": _label_stats(splits.val, cfg.dataset_label_column),
        "test": _label_stats(splits.test, cfg.dataset_label_column),
    }

    default_processed_path = Path(cfg.data_processed_dir) / cfg.processed_filename
    processed_path = Path(getattr(handler, "processed_path", default_processed_path))
    default_metadata_path = processed_path.with_suffix(
        f"{processed_path.suffix}.meta.json"
    )
    processed_metadata_path = Path(
        getattr(handler, "processed_metadata_path", default_metadata_path)
    )

    payload: dict[str, Any] = {
        "force_reprocess": force_reprocess,
        "processed_path": str(processed_path.resolve()),
        "processed_metadata_path": str(processed_metadata_path.resolve()),
        "split_rows": {
            "train": int(len(splits.train)),
            "val": int(len(splits.val)),
            "test": int(len(splits.test)),
        },
        "split_source_distribution": {
            split: counts
            for split, counts in source_distribution.items()
            if counts is not None
        },
        "split_label_stats": {
            split: stats for split, stats in label_stats.items() if stats is not None
        },
    }
    fingerprint = _load_json_file(processed_metadata_path)
    if fingerprint is not None:
        payload["processed_fingerprint"] = fingerprint
    return payload


def build_stage_run_metadata(
    run_data_metadata: Mapping[str, Any] | None,
    stage: TrainingStage,
    train_ds: Any,
    eval_ds: Any | None,
) -> dict[str, Any]:
    """Attach stage-level metadata to the run payload."""
    stage_metadata = dict(run_data_metadata) if run_data_metadata is not None else {}
    stage_metadata["training_stage"] = stage.as_metadata(
        train_rows=dataset_row_count(train_ds),
        eval_rows=dataset_row_count(eval_ds),
    )
    return stage_metadata


def serialize_for_terminal(value: Any) -> Any:
    """Convert nested values into JSON-serializable terminal output."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): serialize_for_terminal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_terminal(item) for item in value]
    return str(value)


def print_training_configuration(
    cfg: Config,
    args: TrainingArguments,
    run_data_metadata: Mapping[str, Any] | None,
) -> None:
    """Print resolved run configuration and hyperparameters when verbose."""
    if not cfg.verbose:
        return

    cfg_payload = serialize_for_terminal(asdict(cfg))
    if isinstance(cfg_payload, dict):
        cfg_payload.pop("hf_token", None)

    training_args_payload: dict[str, Any] = {}
    if hasattr(args, "to_dict"):
        training_args_payload = serialize_for_terminal(args.to_dict())

    payload: dict[str, Any] = {
        "run_id": os.getenv("CODLLM_RUN_ID"),
        "output_dir": cfg.output_dir,
        "config": cfg_payload,
        "training_args": training_args_payload,
    }
    if run_data_metadata is not None:
        payload["dataset"] = serialize_for_terminal(run_data_metadata)

    print("Resolved training setup:")
    print(json.dumps(payload, sort_keys=True, indent=2))
