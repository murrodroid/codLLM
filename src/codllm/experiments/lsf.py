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
) -> GeneratedSubmission:
    """Write generated LSF scripts and env files for an experiment spec."""
    project_path = Path(project_dir).resolve()
    runs = tuple(spec.expanded_runs())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    submission_dir = project_path / output_root / f"{_job_slug(spec.name)}-{timestamp}"
    env_dir = submission_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=False)
    (project_path / profile.log_dir).mkdir(parents=True, exist_ok=True)

    for index, run in enumerate(runs, start=1):
        env_path = env_dir / f"run-{index}.env"
        env_path.write_text(format_env_file(run, spec), encoding="utf-8")

    script_path = submission_dir / "submit.lsf"
    script_path.write_text(
        _render_lsf_script(spec, profile, runs, project_path, env_dir), encoding="utf-8"
    )
    manifest_path = submission_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest_payload(
                spec, profile, runs, submission_dir, script_path, env_dir
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
    )


def _render_lsf_script(
    spec: ExperimentSpec,
    profile: LsfProfile,
    runs: tuple[ExperimentRun, ...],
    project_dir: Path,
    env_dir: Path,
) -> str:
    """Render the generated LSF script used for all runs in a submission."""
    run_count = len(runs)
    job_name = _job_name(profile, spec.name, run_count)
    output_template = profile.output_template or (
        f"{profile.log_dir}/%J_%I.out" if run_count > 1 else f"{profile.log_dir}/%J.out"
    )
    env_dir_ref = _path_for_script(env_dir, project_dir)
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
            f'PROJECT_DIR="${{LSB_SUBCWD:-{project_dir}}}"',
            'RUN_INDEX="${LSB_JOBINDEX:-1}"',
            f'RUN_ENV_FILE="${{CODLLM_RUN_ENV_FILE:-$PROJECT_DIR/{env_dir_ref}/run-${{RUN_INDEX}}.env}}"',
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
            'XDG_CACHE_HOME_DIR="${XDG_CACHE_HOME_DIR:-$RUN_STORAGE_DIR/cache/xdg}"',
            'UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN_STORAGE_DIR/cache/uv}"',
            'UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$RUN_STORAGE_DIR/.venv}"',
            'UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RUN_STORAGE_DIR/python}"',
            'UV_SYNC_LOCK_FILE="${UV_SYNC_LOCK_FILE:-$RUN_STORAGE_DIR/.uv-sync.lock}"',
            f'SYNC_ENV="${{SYNC_ENV:-{1 if profile.sync_env else 0}}}"',
            "",
            'CODLLM_OUTPUT_DIR="${CODLLM_OUTPUT_DIR:-$RUN_STORAGE_DIR/runs}"',
            'CODLLM_DATA_RAW_DIR="${CODLLM_DATA_RAW_DIR:-$PROJECT_DIR/data/raw}"',
            'CODLLM_DATA_PROCESSED_DIR="${CODLLM_DATA_PROCESSED_DIR:-$RUN_STORAGE_DIR/data/processed}"',
            'PYTHONHASHSEED="${PYTHONHASHSEED:-${CODLLM_SEED:-42}}"',
            'CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
            'OMP_NUM_THREADS="${OMP_NUM_THREADS:-${LSB_DJOB_NUMPROC:-1}}"',
            'MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"',
            'TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"',
            "export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE TORCH_HOME",
            "export WANDB_DIR WANDB_CACHE_DIR",
            'export XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"',
            "export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR",
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
            '  "$XDG_CACHE_HOME_DIR" \\',
            '  "$UV_CACHE_DIR" \\',
            '  "$UV_PYTHON_INSTALL_DIR" \\',
            '  "$CODLLM_OUTPUT_DIR" \\',
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
            "  local -a train_cmd=(uv run python -m codllm.training)",
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
            '  local -a inference_cmd=(uv run python -m codllm.inference "$INFERENCE_INPUT_PATH")',
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
            'case "${CODLLM_JOB_COMMAND:-train}" in',
            "  train) run_training ;;",
            "  inference) run_inference ;;",
            '  *) echo "ERROR: unsupported CODLLM_JOB_COMMAND: ${CODLLM_JOB_COMMAND:-}"; exit 1 ;;',
            "esac",
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
        },
        "paths": {
            "submission_dir": str(submission_dir),
            "script_path": str(script_path),
            "env_dir": str(env_dir),
        },
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
