"""Resume-safe callbacks and fixed-horizon scheduling for explicit continuation branches."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from transformers import EarlyStoppingCallback

from codllm.trainer_logging import (
    SigtermSaveCallback,
    StageScopedSeq2SeqTrainer,
    StageScopedTrainer,
)
from codllm import run_markers


class ResumableEarlyStoppingCallback(EarlyStoppingCallback):
    """Restore the non-improvement count while retaining the current explicit patience."""

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Recover checkpoint state without reintroducing the parent's patience setting."""
        super().on_train_begin(args, state, control, **kwargs)
        saved = state.stateful_callbacks.get(type(self).__name__)
        if saved is None:
            saved = state.stateful_callbacks.get("EarlyStoppingCallback", {})
        self.early_stopping_patience_counter = int(
            saved.get("attributes", {}).get("early_stopping_patience_counter", 0)
        )
        state.stateful_callbacks[type(self).__name__] = self.state()
        state.stateful_callbacks.setdefault("TrainerControl", control.state())
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

    def on_evaluate(
        self, args: Any, state: Any, control: Any, metrics: Any, **kwargs: Any
    ) -> None:
        """Count only training-time validation, not final restored-model reevaluations."""
        key = args.metric_for_best_model
        key = key if key.startswith("eval_") else f"eval_{key}"
        if key in metrics:
            super().on_evaluate(args, state, control, metrics, **kwargs)


class ContinuationSigtermSaveCallback(SigtermSaveCallback):
    """Treat a scheduler SIGTERM as an interrupted allocation, never as completed training."""

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Mark a resumable pause before the trainer handles its save/stop flags."""
        control = super().on_step_end(args, state, control, **kwargs)
        if self._signaled:
            run_markers.mark_resume_needed(
                run_markers.run_state_dir(Path(args.output_dir).parent)
            )
        return control


class OriginalScheduleMixin:
    """Keep the parent's update-based cosine decay, clamping it to zero after its horizon."""

    continuation_manifest: dict[str, Any]

    def _continuation_paused(self) -> bool:
        """Distinguish allocation pauses from genuine early stopping or epoch exhaustion."""
        return run_markers.is_resume_needed(
            run_markers.run_state_dir(self.continuation_manifest["destination"])
        )

    def _maybe_log_save_evaluate(self, *args: Any, **kwargs: Any) -> None:
        """Keep wall-time saves even when save-best evaluation would cancel them."""
        paused = self._continuation_paused()
        if paused and not float(self.state.epoch or 0).is_integer():
            self.control.should_evaluate = False
        evaluated = self.control.should_evaluate
        super()._maybe_log_save_evaluate(*args, **kwargs)
        finished = self.state.global_step >= self.state.max_steps or any(
            isinstance(callback, ResumableEarlyStoppingCallback)
            and callback.early_stopping_patience_counter
            >= callback.early_stopping_patience
            for callback in self.callback_handler.callbacks
        )
        if paused and finished:
            run_markers.clear_resume_needed(
                run_markers.run_state_dir(self.continuation_manifest["destination"])
            )
            return
        checkpoint = (
            Path(self.args.output_dir)
            / f"checkpoint-{self.state.global_step}/trainer_state.json"
        )
        if paused and (evaluated or not checkpoint.is_file()):
            trial = kwargs.get("trial", args[3] if len(args) > 3 else None)
            self._save_checkpoint(self.model, trial)

    def _finalize_training(self, *args: Any, **kwargs: Any) -> Any:
        """Do not restore best weights or delete the latest resumable checkpoint during a pause."""
        if not self._continuation_paused():
            return super()._finalize_training(*args, **kwargs)
        load_best, limit = self.args.load_best_model_at_end, self.args.save_total_limit
        self.args.load_best_model_at_end = False
        if limit == 1:
            self.args.save_total_limit = 2
        try:
            return super()._finalize_training(*args, **kwargs)
        finally:
            self.args.load_best_model_at_end = load_best
            self.args.save_total_limit = limit

    def create_scheduler(self, num_training_steps: int, optimizer: Any = None) -> Any:
        """Build the original schedule once; do not stretch it or restart it at each allocation."""
        info = self.continuation_manifest
        original_steps = info["original_max_steps"]
        expected_steps = (
            original_steps * info["target_epochs"] / info["original_epochs"]
        )
        if num_training_steps != expected_steps:
            raise ValueError(
                "Continuation update counts changed; check dataset size, batching, and GPU count."
            )
        first_creation = self.lr_scheduler is None
        scheduler = super().create_scheduler(original_steps, optimizer)
        if first_creation:
            scheduler.lr_lambdas = [
                (lambda step, original=fn: original(min(step, original_steps)))
                for fn in scheduler.lr_lambdas
            ]
        return scheduler


class ContinuationSeq2SeqTrainer(OriginalScheduleMixin, StageScopedSeq2SeqTrainer):
    """Sequence-to-sequence continuation preserving the original learning-rate horizon."""


class ContinuationTrainer(OriginalScheduleMixin, StageScopedTrainer):
    """Classifier continuation preserving the original learning-rate horizon."""
