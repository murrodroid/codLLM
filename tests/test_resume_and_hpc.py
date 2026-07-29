"""Fast unit tests for auto-resume, per-size output dirs, and HPC callbacks."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from codllm import run_markers
from codllm.run_directory import (
    _latest_existing_run_dir,
    _looks_like_run_dir,
    _safe_model_name,
)
from codllm.trainer_logging import (
    SigtermSaveCallback,
    TimeBudgetCallback,
)
from codllm.training.trainer_factory import _has_existing_checkpoint


class TestSafeModelName:
    def test_strips_org_prefix(self) -> None:
        assert _safe_model_name("google/flan-t5-large") == "flan-t5-large"

    def test_replaces_unsafe_characters(self) -> None:
        assert _safe_model_name("foo/bar bee$") == "bar_bee"

    def test_blank_input_yields_default(self) -> None:
        assert _safe_model_name("") == "model"


class TestLooksLikeRunDir:
    def test_run_pattern_match(self) -> None:
        assert _looks_like_run_dir(Path("runs/run-0007"))

    def test_non_run_pattern(self) -> None:
        assert not _looks_like_run_dir(Path("runs/flan-t5-small"))


class TestLatestExistingRunDir:
    def test_returns_none_when_base_missing(self, tmp_path: Path) -> None:
        assert _latest_existing_run_dir(tmp_path / "missing") is None

    def test_returns_none_when_no_run_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-100").mkdir()
        assert _latest_existing_run_dir(tmp_path) is None

    def test_picks_highest_numbered_run(self, tmp_path: Path) -> None:
        for n in (1, 2, 5, 4):
            (tmp_path / f"run-{n:04d}").mkdir()
        latest = _latest_existing_run_dir(tmp_path)
        assert latest is not None
        assert latest.name == "run-0005"


class TestHasExistingCheckpoint:
    def test_false_for_missing_dir(self, tmp_path: Path) -> None:
        assert not _has_existing_checkpoint(str(tmp_path / "missing"))

    def test_false_when_no_checkpoints(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        assert not _has_existing_checkpoint(str(tmp_path))

    def test_true_when_checkpoint_subdir_present(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-2000").mkdir()
        assert _has_existing_checkpoint(str(tmp_path))


class TestTimeBudgetCallback:
    def _fake_args_state_control(self) -> tuple[Any, Any, TrainerControl]:
        # MagicMock stands in for TrainingArguments / TrainerState during pure
        # callback invocations - the callbacks don't read fields from them.
        args = MagicMock(spec=TrainingArguments)
        args.output_dir = None
        return args, MagicMock(spec=TrainerState), TrainerControl()

    def test_no_op_when_budget_disabled(self) -> None:
        callback = TimeBudgetCallback(max_runtime_seconds=0)
        args, state, control = self._fake_args_state_control()
        callback.on_train_begin(args, state, control)
        out = callback.on_step_end(args, state, control)
        assert out.should_training_stop is False
        assert out.should_save is False

    def test_triggers_stop_after_budget_elapsed(self) -> None:
        callback = TimeBudgetCallback(
            max_runtime_seconds=10.0, safety_margin_seconds=2.0
        )
        args, state, control = self._fake_args_state_control()
        callback.on_train_begin(args, state, control)
        # Pretend training started 30 seconds ago - well past 10-2=8s deadline.
        callback._start_time = time.monotonic() - 30
        out = callback.on_step_end(args, state, control)
        assert out.should_training_stop is True
        assert out.should_save is True

    def test_idempotent_after_stop(self) -> None:
        callback = TimeBudgetCallback(
            max_runtime_seconds=5.0, safety_margin_seconds=1.0
        )
        args, state, control = self._fake_args_state_control()
        callback.on_train_begin(args, state, control)
        callback._start_time = time.monotonic() - 60
        callback.on_step_end(args, state, control)
        # Reset control to verify it doesn't re-flip the flag again.
        control = TrainerControl()
        out = callback.on_step_end(args, state, control)
        assert out.should_training_stop is False  # callback._stopping latched

    def test_remaining_seconds_returns_inf_when_disabled(self) -> None:
        callback = TimeBudgetCallback(max_runtime_seconds=0)
        callback.on_train_begin(*self._fake_args_state_control())
        assert callback.remaining_seconds() == float("inf")

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError):
            TimeBudgetCallback(max_runtime_seconds=-1)

    def test_job_start_epoch_anchors_budget_to_job_wall_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 100s budget, 10s margin -> stop at 90s of *job* wall time. Pretend the
        # job started 80s ago (setup), so only ~10s of budget remain.
        callback = TimeBudgetCallback(
            max_runtime_seconds=100.0, safety_margin_seconds=10.0
        )
        monkeypatch.setenv(
            TimeBudgetCallback.JOB_START_EPOCH_ENV, str(time.time() - 80)
        )
        args, state, control = self._fake_args_state_control()
        callback.on_train_begin(args, state, control)
        remaining = callback.remaining_seconds()
        # deadline 90s from job start, ~80s already elapsed -> ~10s left.
        assert 0 < remaining < 20

    def test_job_start_epoch_absent_measures_from_training_start(self) -> None:
        callback = TimeBudgetCallback(
            max_runtime_seconds=100.0, safety_margin_seconds=10.0
        )
        args, state, control = self._fake_args_state_control()
        callback.on_train_begin(args, state, control)
        # No env anchor: budget starts now, so ~90s remain.
        assert 80 < callback.remaining_seconds() <= 90

    def test_writes_resume_marker_on_stop(self, tmp_path: Path) -> None:
        callback = TimeBudgetCallback(
            max_runtime_seconds=10.0,
            safety_margin_seconds=2.0,
            output_dir=str(tmp_path),
        )
        args, state, control = self._fake_args_state_control()
        callback.on_train_begin(args, state, control)
        callback._start_time = time.monotonic() - 30
        callback.on_step_end(args, state, control)
        marker = run_markers.resume_needed_path(tmp_path)
        assert marker.exists()
        assert "max_runtime_seconds=10" in marker.read_text()


class TestRunMarkers:
    def test_state_dir_prefers_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(run_markers.RUN_STATE_DIR_ENV, raising=False)
        assert run_markers.run_state_dir(tmp_path) == tmp_path
        override = tmp_path / "stable"
        monkeypatch.setenv(run_markers.RUN_STATE_DIR_ENV, str(override))
        assert run_markers.run_state_dir(tmp_path / "run-dir") == override

    def test_resume_marker_round_trip(self, tmp_path: Path) -> None:
        assert not run_markers.is_resume_needed(tmp_path)
        run_markers.mark_resume_needed(
            tmp_path, metadata={"safety_margin_seconds": "300"}
        )
        assert run_markers.is_resume_needed(tmp_path)
        assert "safety_margin_seconds=300" in (
            run_markers.resume_needed_path(tmp_path).read_text()
        )
        run_markers.clear_resume_needed(tmp_path)
        assert not run_markers.is_resume_needed(tmp_path)

    def test_mark_complete_clears_stale_resume_and_stamps(self, tmp_path: Path) -> None:
        # A stale resume request from a prior slot must not survive completion.
        run_markers.mark_resume_needed(tmp_path)
        run_markers.mark_training_complete(tmp_path)
        assert not run_markers.is_resume_needed(tmp_path)
        assert run_markers.is_training_complete(tmp_path)
        assert run_markers.final_eval_done_path(tmp_path).exists()

    def test_clear_missing_marker_is_silent(self, tmp_path: Path) -> None:
        run_markers.clear_resume_needed(tmp_path / "does-not-exist")


class TestSigtermSaveCallback:
    def test_flag_triggers_save_and_stop(self) -> None:
        callback = SigtermSaveCallback()
        callback._signaled = True
        control = TrainerControl()
        out = callback.on_step_end(
            MagicMock(spec=TrainingArguments),
            MagicMock(spec=TrainerState),
            control,
        )
        assert out.should_training_stop is True
        assert out.should_save is True

    def test_no_signal_means_no_change(self) -> None:
        callback = SigtermSaveCallback()
        control = TrainerControl()
        out = callback.on_step_end(
            MagicMock(spec=TrainingArguments),
            MagicMock(spec=TrainerState),
            control,
        )
        assert out.should_training_stop is False
        assert out.should_save is False

    def test_handler_install_failure_is_silent(self) -> None:
        """signal.signal raises ValueError off the main thread - we degrade gracefully."""
        callback = SigtermSaveCallback()
        with patch("codllm.trainer_logging.signal.signal", side_effect=ValueError):
            callback.on_train_begin(
                MagicMock(spec=TrainingArguments),
                MagicMock(spec=TrainerState),
                TrainerControl(),
            )
        assert callback._handlers_installed is False


class TestWandbRunIdSidecar:
    def test_round_trip(self, tmp_path: Path) -> None:
        from codllm.wandb_utils import (
            _read_wandb_run_id_sidecar,
            _write_wandb_run_id_sidecar,
        )

        cfg = SimpleNamespace(output_dir=str(tmp_path))
        assert _read_wandb_run_id_sidecar(cfg) is None

        _write_wandb_run_id_sidecar(cfg, "abc123")
        assert _read_wandb_run_id_sidecar(cfg) == "abc123"

    def test_empty_id_does_not_create_file(self, tmp_path: Path) -> None:
        from codllm.wandb_utils import (
            _wandb_run_id_sidecar_path,
            _write_wandb_run_id_sidecar,
        )

        cfg = SimpleNamespace(output_dir=str(tmp_path))
        _write_wandb_run_id_sidecar(cfg, None)
        assert not _wandb_run_id_sidecar_path(cfg).exists()
        _write_wandb_run_id_sidecar(cfg, "")
        assert not _wandb_run_id_sidecar_path(cfg).exists()

    def test_overwrites_only_when_id_changes(self, tmp_path: Path) -> None:
        from codllm.wandb_utils import (
            _wandb_run_id_sidecar_path,
            _write_wandb_run_id_sidecar,
        )

        cfg = SimpleNamespace(output_dir=str(tmp_path))
        _write_wandb_run_id_sidecar(cfg, "abc123")
        first_mtime = _wandb_run_id_sidecar_path(cfg).stat().st_mtime_ns
        time.sleep(0.01)
        _write_wandb_run_id_sidecar(cfg, "abc123")
        second_mtime = _wandb_run_id_sidecar_path(cfg).stat().st_mtime_ns
        assert first_mtime == second_mtime  # short-circuit avoided rewrite

    def test_uses_resume_stable_state_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh scheduler slots should share one W&B run-id sidecar."""
        from codllm.wandb_utils import (
            _read_wandb_run_id_sidecar,
            _wandb_run_id_sidecar_path,
            _write_wandb_run_id_sidecar,
        )

        state_dir = tmp_path / "stable-state"
        monkeypatch.setenv(run_markers.RUN_STATE_DIR_ENV, str(state_dir))
        first_cfg = SimpleNamespace(output_dir=str(tmp_path / "run-100"))
        second_cfg = SimpleNamespace(output_dir=str(tmp_path / "run-200"))

        _write_wandb_run_id_sidecar(first_cfg, "shared123")

        assert _wandb_run_id_sidecar_path(first_cfg) == (state_dir / "wandb_run_id.txt")
        assert _read_wandb_run_id_sidecar(second_cfg) == "shared123"

    def test_reads_legacy_checkpoint_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing runs should migrate from checkpoint-local sidecars."""
        from codllm.wandb_utils import _read_wandb_run_id_sidecar

        output_dir = tmp_path / "run-100"
        output_dir.mkdir()
        (output_dir / "wandb_run_id.txt").write_text("legacy123\n", encoding="utf-8")
        monkeypatch.setenv(
            run_markers.RUN_STATE_DIR_ENV,
            str(tmp_path / "stable-state"),
        )
        cfg = SimpleNamespace(output_dir=str(output_dir))

        assert _read_wandb_run_id_sidecar(cfg) == "legacy123"


class TestPrepareRunOutputDirAutoResume:
    def test_auto_resume_picks_existing_run(self, tmp_path: Path) -> None:
        from codllm.run_directory import prepare_run_output_dir

        # Pre-create an existing run-0001 dir under base.
        base = tmp_path / "runs"
        base.mkdir()
        existing = base / "run-0001"
        existing.mkdir()
        cfg = SimpleNamespace(
            output_dir=str(base),
            hf_model="google/flan-t5-small",
            auto_resume=True,
            per_size_output_dir=False,
        )
        resolved = prepare_run_output_dir(cfg)
        assert resolved == existing

    def test_auto_resume_picks_existing_lsf_array_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resubmitted array cells should rediscover run-JOBID_INDEX checkpoints."""
        from codllm.run_directory import prepare_run_output_dir

        base = tmp_path / "runs"
        base.mkdir()
        existing = base / "run-28888934_3"
        existing.mkdir()
        (base / "run-99999999_2").mkdir()
        monkeypatch.setenv("LSB_JOBINDEX", "3")
        cfg = SimpleNamespace(
            output_dir=str(base),
            hf_model="google/flan-t5-base",
            auto_resume=True,
            per_size_output_dir=False,
        )

        resolved = prepare_run_output_dir(cfg)

        assert resolved == existing

    def test_per_size_inserts_model_subdir(self, tmp_path: Path) -> None:
        from codllm.run_directory import prepare_run_output_dir

        base = tmp_path / "runs"
        base.mkdir()
        cfg = SimpleNamespace(
            output_dir=str(base),
            hf_model="google/flan-t5-large",
            auto_resume=False,
            per_size_output_dir=True,
        )
        resolved = prepare_run_output_dir(cfg)
        assert resolved.parent.name == "flan-t5-large"
        assert resolved.name.startswith("run-")

    def test_existing_run_dir_path_is_used_directly(self, tmp_path: Path) -> None:
        from codllm.run_directory import prepare_run_output_dir

        run_dir = tmp_path / "run-0042"
        run_dir.mkdir()
        cfg = SimpleNamespace(
            output_dir=str(run_dir),
            hf_model="google/flan-t5-small",
            auto_resume=False,
            per_size_output_dir=False,
        )
        resolved = prepare_run_output_dir(cfg)
        assert resolved == run_dir
