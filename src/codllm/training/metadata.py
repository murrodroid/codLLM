from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Mapping

from codllm.config import Config
from codllm.data import DataHandler, DataSplits
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


def _normalized_text_values(dataset: Any, text_column: str) -> list[str] | None:
    """Return normalized text values from a dataframe-like split."""
    if dataset is None or not hasattr(dataset, "columns"):
        return None
    if text_column not in dataset.columns:
        return None
    return [
        " ".join(text.split())
        for text in dataset[text_column].fillna("").astype(str).tolist()
    ]


def _normalized_label_values(dataset: Any, label_column: str) -> list[str] | None:
    """Return stripped label strings from a dataframe-like split."""
    if dataset is None or not hasattr(dataset, "columns"):
        return None
    if label_column not in dataset.columns:
        return None
    return [
        label.strip() for label in dataset[label_column].fillna("").astype(str).tolist()
    ]


def _split_label_codes(label: str, label_separator: str) -> list[str]:
    """Split one label string into non-empty label codes."""
    if label_separator:
        values = label.split(label_separator)
    else:
        values = [label]
    return [value.strip() for value in values if value.strip()]


def _holdout_leakage_stats(cfg: Config, splits: DataSplits) -> dict[str, Any] | None:
    """Summarize holdout overlap with the train split."""
    train_texts = _normalized_text_values(splits.train, cfg.dataset_text_column)
    holdout_texts = _normalized_text_values(splits.holdout, cfg.dataset_text_column)
    train_labels = _normalized_label_values(splits.train, cfg.dataset_label_column)
    holdout_labels = _normalized_label_values(
        splits.holdout,
        cfg.dataset_label_column,
    )
    if (
        train_texts is None
        or holdout_texts is None
        or train_labels is None
        or holdout_labels is None
        or not holdout_texts
    ):
        return None

    train_text_set = set(train_texts)
    train_text_label_pairs = set(zip(train_texts, train_labels))
    input_seen_count = sum(text in train_text_set for text in holdout_texts)
    input_label_pair_seen_count = sum(
        (text, label) in train_text_label_pairs
        for text, label in zip(holdout_texts, holdout_labels)
    )

    train_label_codes: set[str] = set()
    for label in train_labels:
        train_label_codes.update(_split_label_codes(label, cfg.label_separator))

    holdout_label_codes: list[str] = []
    for label in holdout_labels:
        holdout_label_codes.extend(_split_label_codes(label, cfg.label_separator))
    holdout_unique_labels = set(holdout_label_codes)
    label_occurrence_seen_count = sum(
        label in train_label_codes for label in holdout_label_codes
    )
    unique_label_seen_count = len(holdout_unique_labels.intersection(train_label_codes))

    holdout_rows = len(holdout_texts)
    holdout_label_occurrences = len(holdout_label_codes)
    holdout_unique_label_count = len(holdout_unique_labels)
    return {
        "input_seen_count": int(input_seen_count),
        "input_seen_rate": float(input_seen_count / holdout_rows),
        "input_label_pair_seen_count": int(input_label_pair_seen_count),
        "input_label_pair_seen_rate": float(input_label_pair_seen_count / holdout_rows),
        "label_occurrence_seen_count": int(label_occurrence_seen_count),
        "label_occurrence_seen_rate": float(
            label_occurrence_seen_count / holdout_label_occurrences
        )
        if holdout_label_occurrences > 0
        else 0.0,
        "unique_label_seen_count": int(unique_label_seen_count),
        "unique_label_seen_rate": float(
            unique_label_seen_count / holdout_unique_label_count
        )
        if holdout_unique_label_count > 0
        else 0.0,
        "holdout_rows": int(holdout_rows),
        "holdout_label_occurrences": int(holdout_label_occurrences),
        "holdout_unique_labels": int(holdout_unique_label_count),
        "train_unique_labels": int(len(train_label_codes)),
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
        "holdout": _source_distribution(splits.holdout),
        "holdout_eval": _source_distribution(splits.holdout_eval),
    }
    label_stats = {
        "train": _label_stats(splits.train, cfg.dataset_label_column),
        "val": _label_stats(splits.val, cfg.dataset_label_column),
        "test": _label_stats(splits.test, cfg.dataset_label_column),
        "holdout": _label_stats(splits.holdout, cfg.dataset_label_column),
        "holdout_eval": _label_stats(splits.holdout_eval, cfg.dataset_label_column),
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
    holdout_rows = dataset_row_count(splits.holdout)
    if holdout_rows is not None:
        payload["split_rows"]["holdout"] = holdout_rows
    holdout_eval_rows = dataset_row_count(splits.holdout_eval)
    if holdout_eval_rows is not None:
        payload["split_rows"]["holdout_eval"] = holdout_eval_rows
    holdout_leakage = _holdout_leakage_stats(cfg, splits)
    if holdout_leakage is not None:
        payload["holdout_leakage"] = holdout_leakage
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
