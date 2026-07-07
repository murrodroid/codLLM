"""Filesystem markers that coordinate HPC time-budget resume across job slots.

Long training jobs on the wall-time-limited HPC queues cannot finish inside a
single scheduler slot. We split a run's lifecycle into two explicit states,
each represented by a marker file the training process and the generated LSF
script both agree on:

  * ``.resume_needed`` - the slot ended for wall-time reasons but training is
    *not* finished. A checkpoint was saved and the job wants another slot.
  * ``.training_complete`` - training finished for real (epochs exhausted or
    early stopping) and the one-shot final test/holdout/uncertainty evaluation
    has run. No further slots are needed.
  * ``.final_eval_done`` - written alongside ``.training_complete`` as a
    convenience signal that final evaluation has run exactly once.

Markers live in a *state directory*. On HPC the generated LSF script exports
``CODLLM_RUN_STATE_DIR`` to a path that is stable across resubmissions (so the
next slot, which gets a fresh scheduler-allocated run directory, still sees the
markers). When the env var is absent - local runs, tests - the markers fall
back to the run's ``output_dir``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Mapping

RESUME_NEEDED_MARKER = ".resume_needed"
TRAINING_COMPLETE_MARKER = ".training_complete"
FINAL_EVAL_DONE_MARKER = ".final_eval_done"

RUN_STATE_DIR_ENV = "CODLLM_RUN_STATE_DIR"


def run_state_dir(output_dir: str | os.PathLike[str] | None) -> Path:
    """Return the directory used for lifecycle markers.

    ``CODLLM_RUN_STATE_DIR`` wins when set (a stable, resubmission-safe path),
    otherwise the caller's ``output_dir`` is used.
    """
    override = os.getenv(RUN_STATE_DIR_ENV)
    if override and override.strip():
        return Path(override)
    return Path(output_dir or ".")


def resume_needed_path(state_dir: str | os.PathLike[str]) -> Path:
    """Return the ``.resume_needed`` marker path under ``state_dir``."""
    return Path(state_dir) / RESUME_NEEDED_MARKER


def training_complete_path(state_dir: str | os.PathLike[str]) -> Path:
    """Return the ``.training_complete`` marker path under ``state_dir``."""
    return Path(state_dir) / TRAINING_COMPLETE_MARKER


def final_eval_done_path(state_dir: str | os.PathLike[str]) -> Path:
    """Return the ``.final_eval_done`` marker path under ``state_dir``."""
    return Path(state_dir) / FINAL_EVAL_DONE_MARKER


def _write(path: Path, body: str) -> None:
    """Best-effort marker write; markers must never fail a training run."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except OSError:
        return


def _clear(path: Path) -> None:
    """Best-effort marker removal that tolerates an already-absent file."""
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        return


def mark_resume_needed(
    state_dir: str | os.PathLike[str],
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Record that the current slot ran out of wall time before finishing."""
    lines = [f"graceful_exit_at={time.time():.0f}"]
    if metadata:
        lines.extend(f"{key}={value}" for key, value in metadata.items())
    _write(resume_needed_path(state_dir), "\n".join(lines) + "\n")


def clear_resume_needed(state_dir: str | os.PathLike[str]) -> None:
    """Remove any ``.resume_needed`` marker left by a previous slot."""
    _clear(resume_needed_path(state_dir))


def is_resume_needed(state_dir: str | os.PathLike[str]) -> bool:
    """Return True when the current slot requested another slot."""
    return resume_needed_path(state_dir).exists()


def is_training_complete(state_dir: str | os.PathLike[str]) -> bool:
    """Return True when training and final evaluation already finished."""
    return training_complete_path(state_dir).exists()


def mark_training_complete(state_dir: str | os.PathLike[str]) -> None:
    """Record real completion: clear the resume request, stamp completion.

    Clearing ``.resume_needed`` matters because a stale marker from an earlier
    slot would otherwise be misread as "needs another slot".
    """
    clear_resume_needed(state_dir)
    stamp = f"completed_at={time.time():.0f}\n"
    _write(training_complete_path(state_dir), stamp)
    _write(final_eval_done_path(state_dir), stamp)
