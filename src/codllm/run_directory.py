import os
from pathlib import Path
import re

from filelock import FileLock, Timeout

from codllm.config import Config

LOCAL_RUN_DIR_PATTERN = re.compile(r"^run-(\d+)(?:_(\d+))?$")
DEFAULT_RUN_DIR_LOCK_TIMEOUT_SECONDS = 120.0


def run_dir_lock_timeout_seconds() -> float:
    """Return run-directory allocation lock timeout from environment."""
    raw_value = os.getenv("CODLLM_RUN_DIR_LOCK_TIMEOUT_SECONDS")
    if raw_value is None or raw_value.strip() == "":
        return DEFAULT_RUN_DIR_LOCK_TIMEOUT_SECONDS
    try:
        timeout_seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "CODLLM_RUN_DIR_LOCK_TIMEOUT_SECONDS must be a positive float."
        ) from exc
    if timeout_seconds <= 0:
        raise ValueError("CODLLM_RUN_DIR_LOCK_TIMEOUT_SECONDS must be greater than 0.")
    return timeout_seconds


def _next_local_run_number(base_output_dir: Path) -> int:
    """Return the next available local run number in the output root."""
    max_number = 0
    if base_output_dir.exists():
        for path in base_output_dir.iterdir():
            if not path.is_dir():
                continue
            match = LOCAL_RUN_DIR_PATTERN.match(path.name)
            if match is None:
                continue
            max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def resolve_run_output_dir(base_output_dir: str) -> Path:
    """Resolve one run-scoped output directory below the configured root."""
    output_root = Path(base_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run-dir.lock"
    lock_timeout_seconds = run_dir_lock_timeout_seconds()
    lock = FileLock(str(lock_path), timeout=lock_timeout_seconds)
    try:
        with lock:
            hpc_job_id = os.getenv("LSB_JOBID")
            hpc_job_index = os.getenv("LSB_JOBINDEX")
            if hpc_job_id:
                run_id = hpc_job_id
                if hpc_job_index and hpc_job_index not in {"0", ""}:
                    run_id = f"{hpc_job_id}_{hpc_job_index}"
                run_dir = output_root / f"run-{run_id}"
                run_dir.mkdir(parents=True, exist_ok=True)
                return run_dir

            next_run_number = _next_local_run_number(output_root)
            while True:
                run_dir = output_root / f"run-{next_run_number:04d}"
                try:
                    run_dir.mkdir(parents=True, exist_ok=False)
                    return run_dir
                except FileExistsError:
                    next_run_number += 1
    except Timeout as exc:
        raise TimeoutError(
            f"Timed out waiting for run-directory lock '{lock_path}'. "
            "Set CODLLM_RUN_DIR_LOCK_TIMEOUT_SECONDS to a larger value."
        ) from exc


def _safe_model_name(hf_model: str) -> str:
    """Return a filesystem-safe slug for a HuggingFace model id."""
    base = hf_model.split("/")[-1] if "/" in hf_model else hf_model
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_.-")
    return cleaned or "model"


def _looks_like_run_dir(path: Path) -> bool:
    """Return True when path's basename matches the run-* pattern."""
    return LOCAL_RUN_DIR_PATTERN.match(path.name) is not None


def _latest_existing_run_dir(base: Path) -> Path | None:
    """Return the highest-numbered run-NNNN dir under base, or None."""
    if not base.exists():
        return None
    current_job_index = os.getenv("LSB_JOBINDEX")
    if current_job_index in {None, "", "0"}:
        current_job_index = None
    candidates: list[tuple[int, Path]] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = LOCAL_RUN_DIR_PATTERN.match(child.name)
        if match is None:
            continue
        run_job_index = match.group(2)
        if current_job_index is not None and run_job_index != current_job_index:
            continue
        try:
            candidates.append((int(match.group(1)), child))
        except ValueError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def prepare_run_output_dir(cfg: Config) -> Path:
    """Mutate config output_dir to a run-scoped checkpoint root.

    Resolution rules:
      * cfg.output_dir already pointing at a ``run-*`` directory -> use as-is
        (supports caller-specified resume targets).
      * cfg.per_size_output_dir=True -> insert a model-name subdir below the
        configured base before allocating a run id, so size-sweep runs do not
        collide on disk and resume can find the prior run.
      * cfg.auto_resume=True -> reuse the latest run-NNNN under the configured
        base when one exists; otherwise allocate a fresh one.
    """
    base_path = Path(cfg.output_dir)
    if _looks_like_run_dir(base_path) and base_path.exists():
        cfg.output_dir = str(base_path)
        os.environ["CODLLM_RUN_ID"] = base_path.name.removeprefix("run-")
        return base_path

    if cfg.per_size_output_dir:
        base_path = base_path / _safe_model_name(cfg.hf_model)
        base_path.mkdir(parents=True, exist_ok=True)
        cfg.output_dir = str(base_path)

    if cfg.auto_resume:
        existing = _latest_existing_run_dir(base_path)
        if existing is not None:
            cfg.output_dir = str(existing)
            os.environ["CODLLM_RUN_ID"] = existing.name.removeprefix("run-")
            return existing

    run_dir = resolve_run_output_dir(cfg.output_dir)
    cfg.output_dir = str(run_dir)
    os.environ["CODLLM_RUN_ID"] = run_dir.name.removeprefix("run-")
    return run_dir
