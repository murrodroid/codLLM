"""Data preparation helpers."""

from typing import Any

__all__ = [
    "DataHandler",
    "DataSplits",
    "build_and_save_processed_dataset",
    "manipulate_classes",
    "prepare_sequence_classification_dataset",
    "prepare_training_dataset",
    "resolve_training_frames",
    "save_processed_dataset",
    "upsample",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve public data helpers to avoid package import cycles."""
    if name == "DataHandler":
        from codllm.data.handler import DataHandler

        return DataHandler
    if name in {"manipulate_classes", "upsample"}:
        from codllm.data.balancing import manipulate_classes, upsample

        return {"manipulate_classes": manipulate_classes, "upsample": upsample}[name]
    if name in {"DataSplits", "resolve_training_frames"}:
        from codllm.data.splits import DataSplits, resolve_training_frames

        return {
            "DataSplits": DataSplits,
            "resolve_training_frames": resolve_training_frames,
        }[name]
    if name in {"build_and_save_processed_dataset", "save_processed_dataset"}:
        from codllm.data.storage import (
            build_and_save_processed_dataset,
            save_processed_dataset,
        )

        return {
            "build_and_save_processed_dataset": build_and_save_processed_dataset,
            "save_processed_dataset": save_processed_dataset,
        }[name]
    if name in {"prepare_sequence_classification_dataset", "prepare_training_dataset"}:
        from codllm.data.tokenization import (
            prepare_sequence_classification_dataset,
            prepare_training_dataset,
        )

        return {
            "prepare_sequence_classification_dataset": (
                prepare_sequence_classification_dataset
            ),
            "prepare_training_dataset": prepare_training_dataset,
        }[name]
    raise AttributeError(f"module 'codllm.data' has no attribute '{name}'")
