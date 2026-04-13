from typing import Any, Mapping

from transformers import (
    Seq2SeqTrainer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


def _normalize_stage_name(stage_name: str) -> str:
    """Normalize arbitrary stage names to one lowercase token."""
    return stage_name.strip().lower()


def rewrite_metric_key_for_stage(key: str, stage_name: str) -> str:
    """Map trainer metric keys to train/val/test/pretraining categories."""
    if key == "epoch":
        return key

    normalized_stage_name = _normalize_stage_name(stage_name)
    if normalized_stage_name in {"pretrain", "pretraining"}:
        if key.startswith("eval_"):
            return f"pretraining/val/{key.removeprefix('eval_')}"
        if key.startswith("test_"):
            return f"pretraining/test/{key.removeprefix('test_')}"
        if key.startswith("train_"):
            return f"pretraining/{key.removeprefix('train_')}"
        return f"pretraining/{key}"

    if key.startswith("eval_"):
        return f"val/{key.removeprefix('eval_')}"
    if key.startswith("test_"):
        return f"test/{key.removeprefix('test_')}"
    if key.startswith("train_"):
        return f"train/{key.removeprefix('train_')}"
    return f"train/{key}"


def scope_metric_logs_for_stage(
    logs: Mapping[str, Any], stage_name: str
) -> dict[str, Any]:
    """Rewrite one trainer log payload to stage-scoped metric keys."""
    scoped_logs: dict[str, Any] = {}
    for key, value in logs.items():
        scoped_logs[rewrite_metric_key_for_stage(key, stage_name)] = value
    return scoped_logs


class EvaluateEveryNEpochsCallback(TrainerCallback):
    """Skip intermediate eval epochs and evaluate only every Nth epoch."""

    def __init__(self, every_n_epochs: int) -> None:
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1.")
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Run eval only on Nth epochs and always on the final epoch."""
        del args, kwargs
        if not control.should_evaluate:
            return control
        if self.every_n_epochs <= 1:
            return control
        if state.epoch is None:
            return control

        completed_epochs = int(round(state.epoch))
        if completed_epochs < 1:
            return control
        if state.num_train_epochs is None:
            target_epochs = completed_epochs
        else:
            target_epochs = int(round(state.num_train_epochs))

        is_interval_epoch = completed_epochs % self.every_n_epochs == 0
        is_final_epoch = completed_epochs >= target_epochs
        if not (is_interval_epoch or is_final_epoch):
            control.should_evaluate = False
        return control


class StageScopedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer variant that rewrites logged metric keys by stage."""

    def __init__(
        self,
        *args: Any,
        stage_name: str = "train",
        **kwargs: Any,
    ) -> None:
        self.stage_name = stage_name
        super().__init__(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log stage-scoped metrics to callbacks/reporters."""
        scoped_logs = scope_metric_logs_for_stage(logs, self.stage_name)
        super().log(scoped_logs, start_time=start_time)


class StageScopedTrainer(Trainer):
    """Trainer variant that rewrites logged metric keys by stage."""

    def __init__(
        self,
        *args: Any,
        stage_name: str = "train",
        **kwargs: Any,
    ) -> None:
        self.stage_name = stage_name
        super().__init__(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log stage-scoped metrics to callbacks/reporters."""
        scoped_logs = scope_metric_logs_for_stage(logs, self.stage_name)
        super().log(scoped_logs, start_time=start_time)
