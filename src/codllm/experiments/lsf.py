from __future__ import annotations

import json
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from codllm.experiments.specs import (
    ENV_NAME_PATTERN,
    ExperimentRun,
    ExperimentSpec,
    SpecError,
    format_env_file,
    stringify_env_value,
)

JOB_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class LsfProfile:
    """Scheduler and runtime defaults for one LSF submission target."""

    name: str
    queue: str
    wall_time: str
    cores: int
    memory: str
    gpu: str | None = None
    email: str | None = None
    job_name_prefix: str = "codllm"
    log_dir: str = "logs"
    output_template: str | None = None
    span_hosts: int = 1
    env_policy: str = "all"
    storage_folder: str = "/work3/$USER"
    run_storage_dir: str | None = None
    modules: tuple[str, ...] = ()
    extra_resources: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    sync_env: bool = True
    # Hard ceiling on automatic time-budget resubmissions, so a stuck run can
    # never resubmit forever. The submit task derives the actual count from the
    # requested campaign duration and clamps it to this value.
    max_resubmits: int = 30


@dataclass(frozen=True)
class GeneratedSubmission:
    """Paths and commands for a generated LSF submission."""

    spec: ExperimentSpec
    profile: LsfProfile
    runs: tuple[ExperimentRun, ...]
    submission_dir: Path
    script_path: Path
    manifest_path: Path
    env_dir: Path
    project_dir: Path

    def bsub_command(self) -> str:
        """Return the shell command used to submit the generated script."""
        return f"bsub < {shlex.quote(str(self.script_path))}"


def load_lsf_profiles(
    path: Path | str = "hpc/lsf_profiles.toml",
) -> dict[str, LsfProfile]:
    """Load all LSF profiles from a TOML file."""
    profiles_path = Path(path)
    if not profiles_path.exists():
        raise SpecError(f"LSF profiles file does not exist: {profiles_path}.")
    with profiles_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, Mapping):
        raise SpecError("LSF profiles file must contain TOML tables.")
    return {
        name: _build_profile(name, values)
        for name, values in raw.items()
        if isinstance(values, Mapping)
    }


def load_lsf_profile(
    name: str,
    path: Path | str = "hpc/lsf_profiles.toml",
) -> LsfProfile:
    """Load one named LSF profile from a TOML file."""
    profiles = load_lsf_profiles(path)
    try:
        return profiles[name]
    except KeyError as exc:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise SpecError(
            f"Unknown LSF profile '{name}'. Available profiles: {available}."
        ) from exc


def prepare_lsf_submission(
    spec: ExperimentSpec,
    profile: LsfProfile,
    *,
    project_dir: Path | str = ".",
    output_root: Path | str = "jobs/generated",
    env_overrides: Mapping[str, str] | None = None,
) -> GeneratedSubmission:
    """Write generated LSF scripts and env files for an experiment spec.

    ``env_overrides`` are submit-time environment values (e.g. the time-budget
    and resubmission settings derived from ``--duration``) written into every
    run env file with the highest precedence.
    """
    project_path = Path(project_dir).resolve()
    runs = tuple(spec.expanded_runs())
    overrides = dict(env_overrides or {})
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    submission_dir = project_path / output_root / f"{_job_slug(spec.name)}-{timestamp}"
    env_dir = submission_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=False)
    (project_path / profile.log_dir).mkdir(parents=True, exist_ok=True)

    for index, run in enumerate(runs, start=1):
        env_path = env_dir / f"run-{index}.env"
        env_path.write_text(
            format_env_file(run, spec, env_overrides=overrides), encoding="utf-8"
        )

    script_path = submission_dir / "submit.lsf"
    script_path.write_text(
        _render_lsf_script(spec, profile, runs, project_path, env_dir, script_path),
        encoding="utf-8",
    )
    manifest_path = submission_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest_payload(
                spec, profile, runs, submission_dir, script_path, env_dir, overrides
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return GeneratedSubmission(
        spec=spec,
        profile=profile,
        runs=runs,
        submission_dir=submission_dir,
        script_path=script_path,
        manifest_path=manifest_path,
        env_dir=env_dir,
        project_dir=project_path,
    )


def _build_profile(name: str, raw: Mapping[str, Any]) -> LsfProfile:
    """Build a profile from one raw TOML table."""
    queue = _required_string(raw, "queue")
    wall_time = _required_string(raw, "wall_time")
    cores = _required_int(raw, "cores")
    memory = _required_string(raw, "memory")
    if cores < 1:
        raise SpecError("LSF profile 'cores' must be at least 1.")

    return LsfProfile(
        name=name,
        queue=queue,
        wall_time=wall_time,
        cores=cores,
        memory=memory,
        gpu=_optional_string(raw.get("gpu"), "gpu"),
        email=_optional_string(raw.get("email"), "email"),
        job_name_prefix=_optional_string(raw.get("job_name_prefix"), "job_name_prefix")
        or "codllm",
        log_dir=_optional_string(raw.get("log_dir"), "log_dir") or "logs",
        output_template=_optional_string(raw.get("output_template"), "output_template"),
        span_hosts=_optional_int(raw.get("span_hosts"), "span_hosts") or 1,
        env_policy=_optional_string(raw.get("env_policy"), "env_policy") or "all",
        storage_folder=_optional_string(raw.get("storage_folder"), "storage_folder")
        or "/work3/$USER",
        run_storage_dir=_optional_string(raw.get("run_storage_dir"), "run_storage_dir"),
        modules=_optional_string_tuple(raw.get("modules"), "modules"),
        extra_resources=_optional_string_tuple(
            raw.get("extra_resources"), "extra_resources"
        ),
        env=_string_env_table(raw.get("env")),
        sync_env=_optional_bool(raw.get("sync_env"), "sync_env", default=True),
        max_resubmits=_optional_max_resubmits(raw.get("max_resubmits")),
    )


def _optional_max_resubmits(value: object) -> int:
    """Read the optional resubmission ceiling, defaulting to 30."""
    parsed = _optional_int(value, "max_resubmits")
    if parsed is None:
        return 30
    if parsed < 0:
        raise SpecError("LSF profile 'max_resubmits' must be non-negative.")
    return parsed


def _render_lsf_script(
    spec: ExperimentSpec,
    profile: LsfProfile,
    runs: tuple[ExperimentRun, ...],
    project_dir: Path,
    env_dir: Path,
    script_path: Path,
) -> str:
    """Render the generated LSF script used for all runs in a submission."""
    run_count = len(runs)
    job_name = _job_name(profile, spec.name, run_count)
    job_name_base = _job_slug(f"{profile.job_name_prefix}-{spec.name}")[:80]
    output_template = profile.output_template or (
        f"{profile.log_dir}/%J_%I.out" if run_count > 1 else f"{profile.log_dir}/%J.out"
    )
    env_dir_ref = _path_for_script(env_dir, project_dir)
    state_dir_ref = _path_for_script(env_dir.parent / "state", project_dir)
    script_ref = _path_for_script(script_path, project_dir)
    # A resubmission of an array job must re-launch only the current element,
    # not the whole array; override -J with a single-element array spec.
    resubmit_command = (
        f'bsub -J "{job_name_base}[${{RUN_INDEX}}]" < "$SELF_SCRIPT"'
        if run_count > 1
        else 'bsub < "$SELF_SCRIPT"'
    )
    module_lines = "\n".join(
        f"  module load {shlex.quote(module)}" for module in profile.modules
    )
    profile_env_lines = "\n".join(
        _profile_env_default_line(key, value) for key, value in profile.env.items()
    )
    run_storage_line = (
        f'RUN_STORAGE_DIR="${{RUN_STORAGE_DIR:-{profile.run_storage_dir}}}"'
        if profile.run_storage_dir
        else 'RUN_STORAGE_DIR="${RUN_STORAGE_DIR:-$STORAGE_FOLDER/codllm}"'
    )

    lines = [
        "#!/usr/bin/env bash",
        "# ---------------- LSF directives ----------------",
        f'#BSUB -J "{job_name}"',
        f"#BSUB -q {profile.queue}",
        f"#BSUB -W {profile.wall_time}",
        f"#BSUB -n {profile.cores}",
        f'#BSUB -R "span[hosts={profile.span_hosts}]"',
        f'#BSUB -R "rusage[mem={profile.memory}]"',
    ]
    lines.extend(f'#BSUB -R "{resource}"' for resource in profile.extra_resources)
    if profile.gpu:
        lines.append(f'#BSUB -gpu "{profile.gpu}"')
    lines.append(f'#BSUB -env "{profile.env_policy}"')
    if profile.email:
        lines.extend([f"#BSUB -u {profile.email}", "#BSUB -B", "#BSUB -N"])
    lines.extend(
        [
            f"#BSUB -oo {output_template}",
            "# -------------------------------------------------",
            "",
            "set -euo pipefail",
            "",
            "trap 'code=$?;",
            '  printf "ERROR: generated LSF job failed at line %s with exit code %s\\n" "$LINENO" "$code";',
            '  exit "$code"\' ERR',
            "",
            "# Anchor the training time budget to this slot's wall clock, so a"
            " CODLLM_MAX_RUNTIME_SECONDS equal to the LSF -W limit accounts for"
            " setup time too.",
            'export CODLLM_JOB_START_EPOCH="$(date +%s)"',
            "",
            f'PROJECT_DIR="${{LSB_SUBCWD:-{project_dir}}}"',
            'RUN_INDEX="${LSB_JOBINDEX:-1}"',
            f'if [ "{run_count}" = "1" ] && [ "$RUN_INDEX" = "0" ]; then',
            '  RUN_INDEX="1"',
            "fi",
            f'RUN_ENV_FILE="${{CODLLM_RUN_ENV_FILE:-$PROJECT_DIR/{env_dir_ref}/run-${{RUN_INDEX}}.env}}"',
            # Path to this generated script so a time-budget stop can resubmit it.
            f'SELF_SCRIPT="${{CODLLM_SELF_SCRIPT:-$PROJECT_DIR/{script_ref}}}"',
            # Resume-lifecycle markers live in a per-run-index dir that is stable
            # across resubmissions (the scheduler hands each slot a fresh run dir).
            f'CODLLM_RUN_STATE_DIR="${{CODLLM_RUN_STATE_DIR:-$PROJECT_DIR/{state_dir_ref}/run-${{RUN_INDEX}}}}"',
            "export CODLLM_RUN_STATE_DIR",
            'cd "$PROJECT_DIR"',
            "exec 2>&1",
            "",
            'if [ ! -f "$RUN_ENV_FILE" ]; then',
            '  echo "ERROR: generated run env file does not exist: $RUN_ENV_FILE"',
            "  exit 1",
            "fi",
            "",
            'echo "Loading generated run environment: $RUN_ENV_FILE"',
            "set -a",
            'source "$RUN_ENV_FILE"',
            "set +a",
            "",
            f'STORAGE_FOLDER="${{STORAGE_FOLDER:-{profile.storage_folder}}}"',
            run_storage_line,
            'HF_HOME="${HF_HOME:-$RUN_STORAGE_DIR/cache/huggingface}"',
            'HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"',
            'TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"',
            'HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$RUN_STORAGE_DIR/cache/hf_datasets}"',
            'TORCH_HOME="${TORCH_HOME:-$RUN_STORAGE_DIR/cache/torch}"',
            'WANDB_DIR="${WANDB_DIR:-$RUN_STORAGE_DIR/cache/wandb}"',
            'WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"',
            'XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_STORAGE_DIR/cache/xdg}"',
            'UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN_STORAGE_DIR/cache/uv}"',
            'UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$RUN_STORAGE_DIR/.venv}"',
            'UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RUN_STORAGE_DIR/python}"',
            'UV_SYNC_LOCK_FILE="${UV_SYNC_LOCK_FILE:-$RUN_STORAGE_DIR/.uv-sync.lock}"',
            f'SYNC_ENV="${{SYNC_ENV:-{1 if profile.sync_env else 0}}}"',
            'if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$UV_PROJECT_ENVIRONMENT" ]; then',
            "  unset VIRTUAL_ENV",
            "fi",
            "",
            'CODLLM_OUTPUT_DIR="${CODLLM_OUTPUT_DIR:-$RUN_STORAGE_DIR/runs}"',
            'case "$CODLLM_OUTPUT_DIR" in',
            "  /*) ;;",
            '  *) CODLLM_OUTPUT_DIR="$RUN_STORAGE_DIR/$CODLLM_OUTPUT_DIR" ;;',
            "esac",
            'CODLLM_DATA_RAW_DIR="${CODLLM_DATA_RAW_DIR:-$PROJECT_DIR/data/raw}"',
            'CODLLM_DATA_PROCESSED_DIR="${CODLLM_DATA_PROCESSED_DIR:-$RUN_STORAGE_DIR/data/processed}"',
            'PYTHONHASHSEED="${PYTHONHASHSEED:-${CODLLM_SEED:-42}}"',
            'CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
            'OMP_NUM_THREADS="${OMP_NUM_THREADS:-${LSB_DJOB_NUMPROC:-1}}"',
            'MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"',
            'TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"',
            "export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE TORCH_HOME",
            "export WANDB_DIR WANDB_CACHE_DIR",
            "export XDG_CACHE_HOME",
            "export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR",
            "export STORAGE_FOLDER RUN_STORAGE_DIR",
            "export CODLLM_OUTPUT_DIR CODLLM_DATA_RAW_DIR CODLLM_DATA_PROCESSED_DIR",
            "export PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG",
            "export OMP_NUM_THREADS MKL_NUM_THREADS TOKENIZERS_PARALLELISM",
            "export PYTHONUNBUFFERED=1",
            profile_env_lines,
            "",
            'echo "Effective experiment environment:"',
            'echo "  CODLLM_EXPERIMENT_RUN_NAME=${CODLLM_EXPERIMENT_RUN_NAME:-<none>}"',
            'echo "  CODLLM_JOB_COMMAND=${CODLLM_JOB_COMMAND:-train}"',
            'echo "  CODLLM_HF_MODEL=${CODLLM_HF_MODEL:-<Config default>}"',
            'echo "  CODLLM_MODEL_TASK=${CODLLM_MODEL_TASK:-<Config default>}"',
            'echo "  CODLLM_OUTPUT_DIR=$CODLLM_OUTPUT_DIR"',
            'echo "  CODLLM_DATA_RAW_DIR=$CODLLM_DATA_RAW_DIR"',
            'echo "  CODLLM_DATA_PROCESSED_DIR=$CODLLM_DATA_PROCESSED_DIR"',
            'echo "  CODLLM_WANDB_MODE=${CODLLM_WANDB_MODE:-auto}"',
            'echo "  inherited WANDB_MODE=${WANDB_MODE:-<none>}"',
            'echo "  UV_CACHE_DIR=$UV_CACHE_DIR"',
            'echo "  UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"',
            'echo "  UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"',
            "",
            "mkdir -p \\",
            '  "$RUN_STORAGE_DIR" \\',
            '  "$HF_HOME" \\',
            '  "$HF_HUB_CACHE" \\',
            '  "$TRANSFORMERS_CACHE" \\',
            '  "$HF_DATASETS_CACHE" \\',
            '  "$TORCH_HOME" \\',
            '  "$WANDB_DIR" \\',
            '  "$WANDB_CACHE_DIR" \\',
            '  "$XDG_CACHE_HOME" \\',
            '  "$UV_CACHE_DIR" \\',
            '  "$UV_PYTHON_INSTALL_DIR" \\',
            '  "$CODLLM_OUTPUT_DIR" \\',
            '  "$CODLLM_RUN_STATE_DIR" \\',
            '  "$CODLLM_DATA_PROCESSED_DIR"',
            'mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"',
            "",
            'if [ ! -d "$CODLLM_DATA_RAW_DIR" ]; then',
            '  echo "ERROR: CODLLM_DATA_RAW_DIR does not exist: $CODLLM_DATA_RAW_DIR"',
            "  exit 1",
            "fi",
            "",
            "if ! command -v uv >/dev/null 2>&1; then",
            '  echo "ERROR: uv is not available in PATH."',
            "  exit 1",
            "fi",
            "",
            "if command -v module >/dev/null 2>&1; then",
            module_lines or "  true",
            "fi",
            "",
            "if command -v nvidia-smi >/dev/null 2>&1; then",
            "  nvidia-smi || true",
            "fi",
            "",
            'if [ "$SYNC_ENV" = "1" ]; then',
            '  echo "Syncing Python environment with uv."',
            "  if command -v flock >/dev/null 2>&1; then",
            '    flock "$UV_SYNC_LOCK_FILE" uv sync --frozen --no-dev',
            "  else",
            '    echo "WARNING: flock is not available; running uv sync without cross-job locking."',
            "    uv sync --frozen --no-dev",
            "  fi",
            "fi",
            "",
            "is_truthy() {",
            '  case "${1:-}" in',
            "    1|true|TRUE|yes|YES|on|ON) return 0 ;;",
            "    *) return 1 ;;",
            "  esac",
            "}",
            "",
            "run_training() {",
            "  local -a train_cmd=(uv run --no-dev python -m codllm.training)",
            '  if is_truthy "${CODLLM_FORCE_REPROCESS:-${FORCE_REPROCESS:-0}}"; then',
            "    train_cmd+=(--force-reprocess)",
            "  fi",
            '  if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then',
            "    local -a extra_args=()",
            '    read -r -a extra_args <<< "$TRAIN_EXTRA_ARGS"',
            '    train_cmd+=("${extra_args[@]}")',
            "  fi",
            '  echo "Starting training command: ${train_cmd[*]}"',
            '  "${train_cmd[@]}"',
            "}",
            "",
            "run_inference() {",
            '  if [ -z "${INFERENCE_INPUT_PATH:-}" ]; then',
            '    echo "ERROR: INFERENCE_INPUT_PATH must be set for inference jobs."',
            "    exit 1",
            "  fi",
            '  local -a inference_cmd=(uv run --no-dev python -m codllm.inference "$INFERENCE_INPUT_PATH")',
            '  if [ -n "${INFERENCE_OUTPUT_PATH:-}" ]; then',
            '    inference_cmd+=(--output-path "$INFERENCE_OUTPUT_PATH")',
            "  fi",
            '  if is_truthy "${CODLLM_INFERENCE_VALIDATE_REGISTRY:-0}"; then',
            "    inference_cmd+=(--validate-registry)",
            "  fi",
            '  echo "Starting inference command: ${inference_cmd[*]}"',
            '  "${inference_cmd[@]}"',
            "}",
            "",
            "run_wandb_agent() {",
            '  if [ -z "${WANDB_SWEEP_ID:-}" ]; then',
            '    echo "ERROR: WANDB_SWEEP_ID must identify a real W&B sweep."',
            "    exit 1",
            "  fi",
            "  local -a agent_cmd=(",
            "    uv run --no-dev wandb agent --forward-signals --count 1",
            '    "$WANDB_SWEEP_ID"',
            "  )",
            '  echo "Starting one W&B sweep trial: ${agent_cmd[*]}"',
            '  "${agent_cmd[@]}"',
            "}",
            "",
            "# Self-resubmit for another slot when training stopped for wall time.",
            "# Only reached when training exited 0 (a real crash trips the ERR trap",
            "# above and exits non-zero, so crashes never resubmit).",
            "maybe_resubmit_for_resume() {",
            '  if [ "${CODLLM_JOB_COMMAND:-train}" != "train" ]; then',
            "    return 0",
            "  fi",
            '  local state_dir="${CODLLM_RUN_STATE_DIR:-}"',
            '  if [ -z "$state_dir" ]; then',
            "    return 0",
            "  fi",
            '  if [ -f "$state_dir/.training_complete" ]; then',
            '    echo "Training complete; no resubmission needed."',
            "    return 0",
            "  fi",
            '  if [ ! -f "$state_dir/.resume_needed" ]; then',
            "    return 0",
            "  fi",
            '  if ! is_truthy "${CODLLM_AUTO_RESUME:-0}"; then',
            '    echo "WARNING: .resume_needed is set but CODLLM_AUTO_RESUME is disabled;"',
            '    echo "         a resubmitted job would restart from scratch. Not resubmitting."',
            "    return 0",
            "  fi",
            '  local max_resubmits="${CODLLM_MAX_RESUBMITS:-20}"',
            '  local count_file="$state_dir/.resubmit_count"',
            "  local done=0",
            '  if [ -f "$count_file" ]; then',
            '    done="$(cat "$count_file" 2>/dev/null || echo 0)"',
            "  fi",
            '  case "$done" in ""|*[!0-9]*) done=0 ;; esac',
            '  if [ "$done" -ge "$max_resubmits" ]; then',
            '    echo "ERROR: reached CODLLM_MAX_RESUBMITS=$max_resubmits without completing training; not resubmitting."',
            '    : > "$state_dir/.resubmit_exhausted"',
            "    return 0",
            "  fi",
            "  done=$((done + 1))",
            '  printf \'%s\\n\' "$done" > "$count_file"',
            "  if ! command -v bsub >/dev/null 2>&1; then",
            '    echo "WARNING: bsub not found; cannot self-resubmit for resume."',
            "    return 0",
            "  fi",
            '  echo "Resume needed; resubmitting job (resubmission $done/$max_resubmits)."',
            f"  if {resubmit_command}; then",
            '    echo "Resubmitted for resume."',
            "  else",
            '    echo "WARNING: bsub resubmission failed."',
            "  fi",
            "}",
            "",
            'case "${CODLLM_JOB_COMMAND:-train}" in',
            "  train) run_training ;;",
            "  publication-evaluate) uv run --no-dev python -m codllm.evaluation ;;",
            "  publication-baseline) uv run --no-dev python -m codllm.evaluation ;;",
            "  inference) run_inference ;;",
            "  wandb-agent) run_wandb_agent ;;",
            '  *) echo "ERROR: unsupported CODLLM_JOB_COMMAND: ${CODLLM_JOB_COMMAND:-}"; exit 1 ;;',
            "esac",
            "",
            "maybe_resubmit_for_resume",
            "",
            'echo "Job finished successfully."',
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest_payload(
    spec: ExperimentSpec,
    profile: LsfProfile,
    runs: tuple[ExperimentRun, ...],
    submission_dir: Path,
    script_path: Path,
    env_dir: Path,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return serializable manifest data for generated submissions."""
    return {
        "experiment": {
            "name": spec.name,
            "path": str(spec.path),
            "command": spec.command,
            "description": spec.description,
            "run_count": len(runs),
        },
        "profile": {
            "name": profile.name,
            "queue": profile.queue,
            "wall_time": profile.wall_time,
            "cores": profile.cores,
            "memory": profile.memory,
            "gpu": profile.gpu,
            "max_resubmits": profile.max_resubmits,
        },
        "paths": {
            "submission_dir": str(submission_dir),
            "script_path": str(script_path),
            "env_dir": str(env_dir),
        },
        "submit_overrides": dict(env_overrides or {}),
        "runs": [
            {
                "index": index,
                "name": run.name,
                "sweep_values": run.sweep_values,
            }
            for index, run in enumerate(runs, start=1)
        ],
    }


def _job_name(profile: LsfProfile, experiment_name: str, run_count: int) -> str:
    """Return an LSF job name, using array syntax for multi-run submissions."""
    base = _job_slug(f"{profile.job_name_prefix}-{experiment_name}")[:80]
    if run_count > 1:
        return f"{base}[1-{run_count}]"
    return base


def _job_slug(value: str) -> str:
    """Return an LSF-safe job name component."""
    slug = JOB_NAME_PATTERN.sub("-", value).strip("-")
    return slug or "codllm"


def _path_for_script(path: Path, project_dir: Path) -> str:
    """Return a path reference usable from within the generated script."""
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)


def _profile_env_default_line(key: str, value: str) -> str:
    """Return a shell line that sets a profile env default."""
    return f'{key}="${{{key}:-{shlex.quote(value)}}}"\nexport {key}'


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    """Read a required string from a raw profile table."""
    value = raw.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise SpecError(f"LSF profile '{key}' must be a non-empty string.")
    return value


def _required_int(raw: Mapping[str, Any], key: str) -> int:
    """Read a required integer from a raw profile table."""
    value = raw.get(key)
    if not isinstance(value, int):
        raise SpecError(f"LSF profile '{key}' must be an integer.")
    return value


def _optional_string(value: object, key: str) -> str | None:
    """Read an optional string from a raw profile table."""
    if value is None:
        return None
    if not isinstance(value, str) or value.strip() == "":
        raise SpecError(f"LSF profile '{key}' must be a non-empty string when set.")
    return value


def _optional_int(value: object, key: str) -> int | None:
    """Read an optional integer from a raw profile table."""
    if value is None:
        return None
    if not isinstance(value, int):
        raise SpecError(f"LSF profile '{key}' must be an integer when set.")
    return value


def _optional_bool(value: object, key: str, *, default: bool) -> bool:
    """Read an optional boolean from a raw profile table."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SpecError(f"LSF profile '{key}' must be a boolean when set.")
    return value


def _optional_string_tuple(value: object, key: str) -> tuple[str, ...]:
    """Read an optional list of strings from a raw profile table."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpecError(f"LSF profile '{key}' must be a TOML array of strings.")
    return tuple(value)


def _string_env_table(value: object) -> dict[str, str]:
    """Read optional profile environment defaults."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SpecError("LSF profile 'env' must be a TOML table.")
    env: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or not ENV_NAME_PATTERN.match(key):
            raise SpecError(f"Invalid environment variable name in LSF profile: {key}.")
        env[key] = stringify_env_value(raw_value)
    return env
