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


def is_pretraining_stage(stage: TrainingStage) -> bool:
    """Return whether a stage is the masterlist pretraining stage."""
    return stage.name.strip().lower() in {"pretrain", "pretraining"}


def resolved_stage_early_stopping_patience(
    cfg: Config,
    stage: TrainingStage,
) -> int:
    """Return the early-stopping patience for one training stage."""
    if is_pretraining_stage(stage):
        return cfg.resolved_pretrain_early_stopping_patience()
    return cfg.early_stopping_patience


def resolved_stage_load_best_model_at_end(
    cfg: Config,
    stage: TrainingStage,
) -> bool:
    """Return whether one stage reloads its best validation checkpoint."""
    if is_pretraining_stage(stage):
        return cfg.resolved_pretrain_load_best_model_at_end()
    return cfg.load_best_model_at_end


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
    return (
        has_eval_dataset
        and eval_strategy_value == "epoch"
        and is_pretraining_stage(stage)
        and stage.eval_every_n_epochs > 1
    )


def release_stage_trainer_memory(cfg: Config, trainer: Any) -> None:
    """Release trainer-owned state before starting the next stage.

    Frees the stage's optimizer, scheduler, and -- crucially -- the model's
    gradient buffers. Without zeroing grads, the previous stage's per-parameter
    gradient tensors (e.g. ~6 GB for flan-t5-xl) stay resident on the GPU and
    stack on top of the next stage's fresh optimizer + gradients, OOMing the
    pretrain->finetune transfer on large models. set_to_none=True actually
    releases the tensors rather than zeroing them in place.
    """
    if hasattr(trainer, "optimizer"):
        trainer.optimizer = None
    if hasattr(trainer, "lr_scheduler"):
        trainer.lr_scheduler = None
    model = getattr(trainer, "model", None)
    if model is not None and hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    del trainer
    gc.collect()
    if cfg.uses_cuda():
        torch.cuda.empty_cache()
