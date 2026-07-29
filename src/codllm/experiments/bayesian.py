"""W&B Bayesian-sweep helpers for publication screening trials."""

from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from codllm.config import config_from_env
from codllm.experiments.specs import (
    ExperimentSpec,
    SpecError,
    load_experiment_spec,
    stringify_env_value,
)

TRIAL_PARAMETER_ENV = {
    "balance_floor": "CODLLM_BALANCE_FLOOR",
    "multicod_synthetic_ratio": "CODLLM_MULTICOD_SYNTHETIC_RATIO",
    "pretrain_synthetic_ratio": "CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_RATIO",
    "pretrain_target_per_label": "CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL",
    "learning_rate": "CODLLM_LR",
    "scheduler": "CODLLM_LR_SCHEDULER_TYPE",
    "warmup_ratio": "CODLLM_WARMUP_RATIO",
    "base_perturbation_rate": "CODLLM_BASE_PERTURBATION_RATE",
}
PRETRAIN_EPOCHS_PARAMETER = "pretrain_epochs"
TRIAL_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def load_trial_parameters(path: Path | str) -> dict[str, Any]:
    """Load one W&B-generated JSON parameter mapping."""
    parameter_path = Path(path)
    try:
        payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(
            f"Could not load Bayesian trial parameters from {parameter_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise SpecError("Bayesian trial parameters must be a JSON object.")
    return payload


def validate_trial_parameters(parameters: Mapping[str, Any]) -> None:
    """Validate the allowlisted publication search parameters."""
    allowed = set(TRIAL_PARAMETER_ENV) | {PRETRAIN_EPOCHS_PARAMETER}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise SpecError(
            "Unsupported Bayesian trial parameter(s): "
            f"{', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}."
        )

    _validate_integer(parameters, "balance_floor", minimum=0)
    _validate_ratio(parameters, "multicod_synthetic_ratio")
    _validate_integer(parameters, PRETRAIN_EPOCHS_PARAMETER, minimum=0)
    _validate_ratio(parameters, "pretrain_synthetic_ratio")
    _validate_integer(parameters, "pretrain_target_per_label", minimum=1)
    _validate_float(parameters, "learning_rate", minimum=0.0, strict_minimum=True)
    _validate_float(parameters, "warmup_ratio", minimum=0.0, maximum=1.0)
    _validate_ratio(parameters, "base_perturbation_rate")

    scheduler = parameters.get("scheduler")
    if scheduler is not None and scheduler not in {"cosine", "constant"}:
        raise SpecError("scheduler must be either 'cosine' or 'constant'.")


def build_trial_environment(
    base_config: Path | str,
    parameters: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge sampled values into one single-run publication base environment."""
    validate_trial_parameters(parameters)
    spec = load_experiment_spec(base_config)
    runs = spec.expanded_runs()
    if len(runs) != 1:
        raise SpecError(
            "A Bayesian base config must expand to exactly one run; "
            f"{spec.path} expands to {len(runs)}."
        )

    env = runs[0].env_with_runtime_metadata(spec)
    for parameter_name, env_name in TRIAL_PARAMETER_ENV.items():
        if parameter_name in parameters:
            env[env_name] = stringify_env_value(parameters[parameter_name])

    if PRETRAIN_EPOCHS_PARAMETER in parameters:
        pretrain_epochs = int(parameters[PRETRAIN_EPOCHS_PARAMETER])
        env["CODLLM_PRETRAIN_ENABLED"] = "1" if pretrain_epochs > 0 else "0"
        if pretrain_epochs > 0:
            env["CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS"] = str(pretrain_epochs)

    runtime_env = os.environ if environ is None else environ
    trial_id = _trial_id(runtime_env)
    output_root = Path(env.get("CODLLM_OUTPUT_DIR", "checkpoints/publication/bayesian"))
    run_storage_dir = runtime_env.get("RUN_STORAGE_DIR")
    if not output_root.is_absolute() and run_storage_dir:
        output_root = Path(run_storage_dir) / output_root
    env["CODLLM_OUTPUT_DIR"] = str(output_root / f"trial-{trial_id}")
    env["CODLLM_AUTO_RESUME"] = "0"
    env["CODLLM_WANDB_RUN_NAME"] = f"publication-bayes-{trial_id}"
    env["CODLLM_EXPERIMENT_RUN_NAME"] = f"publication-bayes-{trial_id}"
    env["CODLLM_EXPERIMENT_SWEEP_VALUES"] = json.dumps(
        dict(parameters),
        ensure_ascii=True,
        sort_keys=True,
    )
    return env


def build_wandb_agent_spec(
    sweep_id: str,
    agents: int,
    *,
    source_path: Path | str = "runs/publication/bayesian_sweep.yaml",
) -> ExperimentSpec:
    """Build an in-memory LSF array spec with one W&B trial per agent."""
    normalized_sweep_id = sweep_id.strip()
    if not normalized_sweep_id:
        raise SpecError("A real W&B sweep ID is required.")
    if agents < 1:
        raise SpecError("agents must be at least 1.")
    return ExperimentSpec(
        path=Path(source_path).resolve(),
        name="publication_bayesian_agents",
        command="wandb-agent",
        description="One single-slot W&B Bayesian trial per LSF array element.",
        env={
            "WANDB_SWEEP_ID": normalized_sweep_id,
            "CODLLM_OUTPUT_DIR": "checkpoints/publication/bayesian/agents",
            "CODLLM_AUTO_RESUME": "0",
        },
        sweep={
            "CODLLM_BAYES_AGENT_INDEX": tuple(
                str(index) for index in range(1, agents + 1)
            )
        },
    )


def run_trial(
    base_config: Path | str,
    parameters: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Run one mapped Bayesian trial through the normal training pipeline."""
    trial_env = build_trial_environment(
        base_config,
        parameters,
        environ=environ,
    )
    with _temporary_environ(trial_env):
        from codllm.training.pipeline import train

        _, _, splits = train(cfg=config_from_env(), force_reprocess=False)
    print(
        "Bayesian trial completed. "
        f"train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the W&B sweep trial wrapper arguments."""
    parser = argparse.ArgumentParser(
        description="Map one W&B Bayesian sample into a codLLM publication run."
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Single-run TOML protocol inherited by every Bayesian trial.",
    )
    parser.add_argument(
        "--parameters-json",
        required=True,
        help="JSON file generated by the W&B ${args_json_file} command macro.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the mapped environment without training.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run or preview one W&B-selected publication trial."""
    args = parse_args(argv)
    parameters = load_trial_parameters(args.parameters_json)
    if args.dry_run:
        print(
            json.dumps(
                build_trial_environment(args.base_config, parameters),
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_trial(args.base_config, parameters)


def _trial_id(environ: Mapping[str, str]) -> str:
    """Return a filesystem-safe ID unique to one W&B or LSF trial."""
    raw_id = environ.get("WANDB_RUN_ID")
    if not raw_id:
        job_id = environ.get("LSB_JOBID", "manual")
        job_index = environ.get("LSB_JOBINDEX")
        raw_id = f"{job_id}_{job_index}" if job_index else job_id
    normalized = TRIAL_ID_PATTERN.sub("-", raw_id).strip("-.")
    return normalized or "manual"


def _validate_integer(
    parameters: Mapping[str, Any],
    name: str,
    *,
    minimum: int,
) -> None:
    """Validate one optional integer parameter."""
    if name not in parameters:
        return
    value = parameters[name]
    is_integer = isinstance(value, int) or (
        isinstance(value, float) and value.is_integer()
    )
    if isinstance(value, bool) or not is_integer or value < minimum:
        raise SpecError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )


def _validate_ratio(parameters: Mapping[str, Any], name: str) -> None:
    """Validate one optional ratio parameter in the closed unit interval."""
    _validate_float(parameters, name, minimum=0.0, maximum=1.0)


def _validate_float(
    parameters: Mapping[str, Any],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> None:
    """Validate one optional numeric parameter."""
    if name not in parameters:
        return
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{name} must be numeric.")
    if value < minimum or (strict_minimum and value == minimum):
        comparison = "greater than" if strict_minimum else "at least"
        raise SpecError(f"{name} must be {comparison} {minimum}.")
    if maximum is not None and value > maximum:
        raise SpecError(f"{name} must be at most {maximum}.")


@contextmanager
def _temporary_environ(updates: Mapping[str, str]) -> Iterator[None]:
    """Apply trial overrides while preserving the caller's environment."""
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
