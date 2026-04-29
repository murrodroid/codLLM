from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any

import torch

from codllm.config import Config


@dataclass(frozen=True, slots=True)
class TrainingStage:
    """Configuration for one training stage."""

    name: str
    output_dir: str
    num_train_epochs: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    eval_every_n_epochs: int = 1

    def as_metadata(
        self,
        train_rows: int | None = None,
        eval_rows: int | None = None,
    ) -> dict[str, Any]:
        """Return a serializable representation of the stage."""
        payload: dict[str, Any] = {
            "name": self.name,
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "eval_every_n_epochs": self.eval_every_n_epochs,
            "lr_scheduler_type": self.lr_scheduler_type,
        }
        if train_rows is not None or eval_rows is not None:
            payload["rows"] = {
                "train": train_rows,
                "eval": eval_rows,
            }
        return payload


def build_train_stage(cfg: Config) -> TrainingStage:
    """Build the default single-stage fine-tuning configuration."""
    return TrainingStage(
        name="train",
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
    )


def build_pretraining_stage(cfg: Config) -> TrainingStage:
    """Build the pretraining stage configuration."""
    pretrain_learning_rate = (
        cfg.pretrain_learning_rate if cfg.pretrain_learning_rate is not None else cfg.lr
    )
    return TrainingStage(
        name="pretrain",
        output_dir=str(Path(cfg.output_dir) / "pretrain"),
        num_train_epochs=cfg.pretrain_num_train_epochs,
        learning_rate=pretrain_learning_rate,
        warmup_ratio=cfg.pretrain_warmup_ratio,
        lr_scheduler_type=cfg.pretrain_lr_scheduler_type,
        eval_every_n_epochs=cfg.pretrain_eval_every_n_epochs,
    )


def build_finetune_stage(cfg: Config) -> TrainingStage:
    """Build the fine-tuning stage used after pretraining."""
    return TrainingStage(
        name="finetune",
        output_dir=str(Path(cfg.output_dir) / "finetune"),
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
    )


def should_apply_eval_interval_callback(
    stage: TrainingStage,
    eval_strategy_value: str,
    has_eval_dataset: bool,
) -> bool:
    """Return whether the N-epoch eval callback should be attached."""
    normalized_stage_name = stage.name.strip().lower()
    is_pretraining_stage = normalized_stage_name in {"pretrain", "pretraining"}
    return (
        has_eval_dataset
        and eval_strategy_value == "epoch"
        and is_pretraining_stage
        and stage.eval_every_n_epochs > 1
    )


def release_stage_trainer_memory(cfg: Config, trainer: Any) -> None:
    """Release trainer-owned state before starting the next stage."""
    if hasattr(trainer, "optimizer"):
        trainer.optimizer = None
    if hasattr(trainer, "lr_scheduler"):
        trainer.lr_scheduler = None
    del trainer
    gc.collect()
    if cfg.uses_cuda():
        torch.cuda.empty_cache()
