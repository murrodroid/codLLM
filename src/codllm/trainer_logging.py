import logging
import os
import signal
import time
from typing import Any, Mapping

from transformers import (
    Seq2SeqTrainer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from codllm import run_markers
from codllm.metrics import (
    reset_metric_artifact_scope,
    reset_metric_source_ids,
    set_metric_artifact_scope,
    set_metric_source_ids,
)

logger = logging.getLogger(__name__)


def _normalize_stage_name(stage_name: str) -> str:
    """Normalize arbitrary stage names to one lowercase token."""
    return stage_name.strip().lower()


def rewrite_metric_key_for_stage(key: str, stage_name: str) -> str:
    """Map trainer metric keys to stage-scoped logging categories."""
    if key == "epoch":
        return key
    if key.startswith("holdout_full_"):
        return f"holdout/full/{key.removeprefix('holdout_full_')}"
    if key.startswith("holdout_sample_"):
        return f"holdout/sample/{key.removeprefix('holdout_sample_')}"
    if key.startswith("holdout_val_"):
        return f"holdout/val/{key.removeprefix('holdout_val_')}"
    if key.startswith("holdout_test_"):
        return f"holdout/test/{key.removeprefix('holdout_test_')}"
    if key.startswith("holdout_"):
        return f"holdout/{key.removeprefix('holdout_')}"

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
        scoped_key = rewrite_metric_key_for_stage(key, stage_name)
        scoped_logs[scoped_key] = value
        if key.startswith("holdout_full_"):
            scoped_logs[f"holdout/{key.removeprefix('holdout_full_')}"] = value
        elif key.startswith("holdout_sample_"):
            scoped_logs[f"holdout/val/{key.removeprefix('holdout_sample_')}"] = value
        elif key.startswith("holdout_val_"):
            scoped_logs[f"holdout/sample/{key.removeprefix('holdout_val_')}"] = value
        elif key.startswith("holdout_test_"):
            continue
        elif key.startswith("holdout_"):
            scoped_logs[f"holdout/full/{key.removeprefix('holdout_')}"] = value
    return scoped_logs


def _scope_for_metric_key_prefix(metric_key_prefix: str, stage_name: str) -> str:
    """Return the scoped W&B namespace for one Trainer metric key prefix."""
    scoped_metric_key = rewrite_metric_key_for_stage(
        f"{metric_key_prefix}_metric",
        stage_name,
    )
    return scoped_metric_key.removesuffix("/metric")


def _metric_key_prefix_from_evaluate_call(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> str:
    """Extract Trainer.evaluate metric_key_prefix from positional or keyword args."""
    if "metric_key_prefix" in kwargs:
        return str(kwargs["metric_key_prefix"])
    if len(args) >= 3:
        return str(args[2])
    return "eval"


def _eval_dataset_from_evaluate_call(
    self_eval_dataset: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Resolve the eval dataset Trainer.evaluate will iterate, matching its signature."""
    if "eval_dataset" in kwargs:
        eval_dataset = kwargs["eval_dataset"]
    elif args:
        eval_dataset = args[0]
    else:
        eval_dataset = None
    return eval_dataset if eval_dataset is not None else self_eval_dataset


def _resolve_source_ids(eval_dataset: Any) -> list[str] | None:
    """Return source ids registered on the eval dataset, when available."""
    if eval_dataset is None:
        return None
    source_ids = getattr(eval_dataset, "source_ids", None)
    if source_ids is None:
        return None
    return list(source_ids)


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


class HoldoutEvaluationCallback(TrainerCallback):
    """Run sampled hold-out evaluation during training."""

    def __init__(
        self,
        evaluate_per: str,
        eval_dataset: Any,
        eval_steps: int,
        baseline_eval_dataset: Any | None = None,
        metric_key_prefix: str = "holdout_sample",
    ) -> None:
        normalized = evaluate_per.strip().lower()
        if normalized not in {"epoch", "steps"}:
            raise ValueError("evaluate_per must be 'epoch' or 'steps'.")
        if eval_steps < 1:
            raise ValueError("eval_steps must be at least 1.")
        self.evaluate_per = normalized
        self.eval_dataset = eval_dataset
        self.baseline_eval_dataset = baseline_eval_dataset
        self.eval_steps = eval_steps
        self.metric_key_prefix = metric_key_prefix
        self.trainer: Any | None = None
        self._last_step: int | None = None

    def attach_trainer(self, trainer: Any) -> None:
        """Attach the trainer instance used to run evaluation."""
        self.trainer = trainer

    def _evaluate(self, *, include_baseline: bool) -> None:
        """Run one hold-out evaluation pass when a trainer is attached."""
        if self.trainer is None:
            return
        if include_baseline and self.baseline_eval_dataset is not None:
            self.trainer.evaluate(
                eval_dataset=self.baseline_eval_dataset,
                metric_key_prefix="eval",
            )
        self.trainer.evaluate(
            eval_dataset=self.eval_dataset,
            metric_key_prefix=self.metric_key_prefix,
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Evaluate at epoch boundaries when configured."""
        del args, state, kwargs
        if self.evaluate_per == "epoch":
            native_eval_scheduled = control.should_evaluate
            self._evaluate(include_baseline=not native_eval_scheduled)
            if native_eval_scheduled:
                # Our manual evaluate() flips should_evaluate to False
                # (CallbackHandler.on_evaluate), which would suppress the
                # Trainer's own eval that sets best_metric and drives
                # load_best_model_at_end + early stopping. Re-arm it.
                control.should_evaluate = True
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Evaluate at eval-step boundaries when configured."""
        del args, kwargs
        if self.evaluate_per != "steps":
            return control
        global_step = int(getattr(state, "global_step", 0) or 0)
        if global_step < 1 or global_step == self._last_step:
            return control
        if global_step % self.eval_steps == 0:
            self._last_step = global_step
            native_eval_scheduled = control.should_evaluate
            self._evaluate(include_baseline=not native_eval_scheduled)
            if native_eval_scheduled:
                control.should_evaluate = True
        return control


class TimeBudgetCallback(TrainerCallback):
    """Stop training gracefully before an HPC wall-time limit kicks in.

    On every step end it checks elapsed time against
    ``max_runtime_seconds - safety_margin_seconds``; when that budget is
    exhausted we set ``should_save`` and ``should_training_stop`` so the trainer
    writes a clean checkpoint and exits before the scheduler sends SIGKILL.

    The budget is normally measured from the start of training, but when the
    launcher exports ``CODLLM_JOB_START_EPOCH`` (the generated LSF script does)
    it is instead measured from the start of the *job*, so setup time counts too
    and ``max_runtime_seconds`` can be set directly to the scheduler ``-W``
    limit.

    Also drops a ``.resume_needed`` marker file at ``output_dir`` when the
    budget triggers, so the generated LSF script can distinguish a graceful
    "this run wants more wall time" exit from an actual crash and resubmit the
    job. ``output_dir`` here is the resume-stable run state directory (see
    :mod:`codllm.run_markers`), which the trainer factory resolves for us.
    """

    RESUME_MARKER_FILENAME = run_markers.RESUME_NEEDED_MARKER
    JOB_START_EPOCH_ENV = "CODLLM_JOB_START_EPOCH"

    def __init__(
        self,
        max_runtime_seconds: float,
        safety_margin_seconds: float = 300.0,
        output_dir: str | None = None,
    ) -> None:
        if max_runtime_seconds < 0:
            raise ValueError("max_runtime_seconds must be non-negative.")
        if safety_margin_seconds < 0:
            raise ValueError("safety_margin_seconds must be non-negative.")
        self.max_runtime_seconds = float(max_runtime_seconds)
        self.safety_margin_seconds = float(safety_margin_seconds)
        self.output_dir = output_dir
        self._start_time: float | None = None
        self._stopping = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Record the budget start timestamp (job start when available)."""
        del state, kwargs
        self._start_time = time.monotonic()
        # When the launcher recorded a job start, shift the reference back by
        # the wall time already spent this slot so the budget covers setup too.
        job_elapsed = self._job_elapsed_seconds()
        if job_elapsed is not None:
            self._start_time -= job_elapsed
        self._stopping = False
        # Use the trainer's output_dir if we weren't given one at construction.
        if self.output_dir is None and args is not None:
            self.output_dir = getattr(args, "output_dir", None)
        return control

    @classmethod
    def _job_elapsed_seconds(cls) -> float | None:
        """Return wall seconds since job start, or None when not anchored."""
        raw = os.getenv(cls.JOB_START_EPOCH_ENV)
        if not raw:
            return None
        try:
            started = float(raw)
        except ValueError:
            return None
        elapsed = time.time() - started
        return elapsed if elapsed >= 0 else None

    def remaining_seconds(self) -> float:
        """Return seconds remaining in the training budget, or +inf when disabled."""
        if self._start_time is None or self.max_runtime_seconds <= 0:
            return float("inf")
        elapsed = time.monotonic() - self._start_time
        deadline = self.max_runtime_seconds - self.safety_margin_seconds
        return float(deadline - elapsed)

    def _write_resume_marker(self) -> None:
        """Drop a marker file so resubmit logic can distinguish graceful exit."""
        if not self.output_dir:
            return
        run_markers.mark_resume_needed(
            self.output_dir,
            metadata={
                "max_runtime_seconds": f"{self.max_runtime_seconds:.0f}",
                "safety_margin_seconds": f"{self.safety_margin_seconds:.0f}",
            },
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Trigger a graceful save+stop when the budget is about to expire."""
        del args, state, kwargs
        if self.max_runtime_seconds <= 0 or self._stopping:
            return control
        if self.remaining_seconds() <= 0:
            logger.warning(
                "Time budget exhausted (max_runtime=%.0fs, margin=%.0fs); "
                "saving and stopping.",
                self.max_runtime_seconds,
                self.safety_margin_seconds,
            )
            control.should_save = True
            control.should_training_stop = True
            self._stopping = True
            self._write_resume_marker()
        return control


class SigtermSaveCallback(TrainerCallback):
    """Mark a graceful save+stop when SIGTERM arrives.

    Works alongside HF's frequent step-saves: routine saves protect against
    SIGKILL, this callback handles the polite SIGTERM the HPC scheduler
    typically sends ~30s before SIGKILL.
    """

    def __init__(self) -> None:
        self._signaled = False
        self._handlers_installed = False
        self._previous_handler: Any = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Install a SIGTERM handler that flips the graceful-stop flag."""
        del args, state, kwargs
        try:
            self._previous_handler = signal.signal(signal.SIGTERM, self._on_signal)
            self._handlers_installed = True
        except (ValueError, OSError):
            # signal.signal raises ValueError when not on the main thread, and
            # Windows builds may not deliver SIGTERM; degrade silently.
            self._handlers_installed = False
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Restore the previous SIGTERM handler so we leave a clean process state."""
        del args, state, kwargs
        if self._handlers_installed:
            try:
                signal.signal(signal.SIGTERM, self._previous_handler)
            except (ValueError, OSError):
                pass
            self._handlers_installed = False
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """If a SIGTERM was observed, request a graceful save+stop."""
        del args, state, kwargs
        if self._signaled:
            control.should_save = True
            control.should_training_stop = True
        return control

    def _on_signal(self, signum: int, frame: Any) -> None:
        """SIGTERM handler: flip the flag so the next step boundary saves+exits."""
        del signum, frame
        self._signaled = True
        logger.warning("SIGTERM received; will save and stop at next step boundary.")


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

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluate with a stage-scoped context for metric artifact logging."""
        metric_key_prefix = _metric_key_prefix_from_evaluate_call(args, kwargs)
        scope_token = set_metric_artifact_scope(
            _scope_for_metric_key_prefix(metric_key_prefix, self.stage_name)
        )
        eval_dataset = _eval_dataset_from_evaluate_call(self.eval_dataset, args, kwargs)
        source_ids_token = set_metric_source_ids(_resolve_source_ids(eval_dataset))
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            reset_metric_source_ids(source_ids_token)
            reset_metric_artifact_scope(scope_token)


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

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluate with a stage-scoped context for metric artifact logging."""
        metric_key_prefix = _metric_key_prefix_from_evaluate_call(args, kwargs)
        scope_token = set_metric_artifact_scope(
            _scope_for_metric_key_prefix(metric_key_prefix, self.stage_name)
        )
        eval_dataset = _eval_dataset_from_evaluate_call(self.eval_dataset, args, kwargs)
        source_ids_token = set_metric_source_ids(_resolve_source_ids(eval_dataset))
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            reset_metric_source_ids(source_ids_token)
            reset_metric_artifact_scope(scope_token)
