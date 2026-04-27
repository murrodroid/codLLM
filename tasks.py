from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from invoke import Collection, Context, Exit, task

from codllm.experiments import (
    ExperimentRun,
    ExperimentSpec,
    GeneratedSubmission,
    LsfProfile,
    SpecError,
    list_experiment_specs,
    load_experiment_spec,
    load_lsf_profile,
    load_lsf_profiles,
    prepare_lsf_submission,
)

DEFAULT_EXPERIMENT_ROOT = Path("runs")
DEFAULT_PROFILE_PATH = Path("hpc/lsf_profiles.toml")
LSF_USER_EMAILS = {
    "lucas": "s234805@dtu.dk",
    "elias": "s234854@dtu.dk",
}


@task
def sync(ctx: Context) -> None:
    """Synchronize the Python environment from uv.lock."""
    ctx.run("uv sync", pty=True)


@task
def train(
    ctx: Context,
    config: str | None = None,
    sweep_index: int = 1,
    force_reprocess: bool | None = None,
    dry_run: bool = False,
) -> None:
    """Run training locally from the current env or one TOML experiment spec."""
    command = "uv run python -m codllm.training"
    env = os.environ.copy()
    if config is not None:
        spec = load_experiment_spec(config)
        if spec.command != "train":
            raise Exit(
                f"Task 'train' only supports command='train' specs, got {spec.command!r}.",
                code=2,
            )
        run = _select_run(spec, sweep_index)
        env.update(run.env_with_runtime_metadata(spec))
        effective_force_reprocess = (
            run.force_reprocess if force_reprocess is None else force_reprocess
        )
    else:
        effective_force_reprocess = bool(force_reprocess)

    if effective_force_reprocess:
        command = f"{command} --force-reprocess"
    if dry_run:
        print(command)
        return
    ctx.run(command, env=env, pty=True)


@task(name="list")
def experiments_list(ctx: Context, root: str = str(DEFAULT_EXPERIMENT_ROOT)) -> None:
    """List TOML experiment specifications."""
    del ctx
    for path in list_experiment_specs(root):
        try:
            spec = load_experiment_spec(path)
        except SpecError as exc:
            print(f"{path}: invalid ({exc})")
            continue
        description = f" - {spec.description}" if spec.description else ""
        print(
            f"{path}: {spec.name} [{spec.command}] runs={len(spec.expanded_runs())}{description}"
        )


@task(name="plan")
def experiments_plan(ctx: Context, config: str, profile: str | None = None) -> None:
    """Show the concrete runs produced by one experiment specification."""
    del ctx
    spec = load_experiment_spec(config)
    print(_format_spec_summary(spec))
    if profile is not None:
        lsf_profile = load_lsf_profile(profile, DEFAULT_PROFILE_PATH)
        print(
            f"LSF profile: {lsf_profile.name} queue={lsf_profile.queue} "
            f"wall_time={lsf_profile.wall_time} cores={lsf_profile.cores}"
        )
    for index, run in enumerate(spec.expanded_runs(), start=1):
        sweep = _format_sweep(run)
        print(f"{index:>3}. {run.name}{sweep}")


@task(name="profiles")
def hpc_profiles(ctx: Context, profiles: str = str(DEFAULT_PROFILE_PATH)) -> None:
    """List configured LSF profiles."""
    del ctx
    for name, profile in load_lsf_profiles(profiles).items():
        gpu = f" gpu={profile.gpu}" if profile.gpu else ""
        print(
            f"{name}: queue={profile.queue} wall_time={profile.wall_time} cores={profile.cores}{gpu}"
        )


@task(name="storage")
def hpc_storage(ctx: Context) -> None:
    """Print uv/cache paths and warn when they are outside RUN_STORAGE_DIR."""
    run_storage_dir = os.getenv("RUN_STORAGE_DIR")
    paths = {
        "RUN_STORAGE_DIR": run_storage_dir,
        "UV_CACHE_DIR": os.getenv("UV_CACHE_DIR"),
        "UV_PROJECT_ENVIRONMENT": os.getenv("UV_PROJECT_ENVIRONMENT"),
        "UV_PYTHON_INSTALL_DIR": os.getenv("UV_PYTHON_INSTALL_DIR"),
        "HF_HOME": os.getenv("HF_HOME"),
        "TORCH_HOME": os.getenv("TORCH_HOME"),
        "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME"),
        "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV"),
    }
    uv_cache = ctx.run("uv cache dir", hide=True).stdout.strip()
    paths["uv cache dir"] = uv_cache

    for key, value in paths.items():
        print(f"{key}={value or '<unset>'}")

    if run_storage_dir is None:
        print(
            "WARNING: RUN_STORAGE_DIR is unset. Source hpc/env.sh before running uv on HPC."
        )
        return

    storage_root = Path(run_storage_dir).expanduser()
    for key, value in paths.items():
        if key == "RUN_STORAGE_DIR" or value is None:
            continue
        path = Path(value).expanduser()
        if not _is_relative_to(path, storage_root):
            print(f"WARNING: {key} is outside RUN_STORAGE_DIR: {value}")


@task(name="submit")
def hpc_submit(
    ctx: Context,
    config: str,
    profile: str = "h100-10h",
    user: str | None = None,
    profiles: str = str(DEFAULT_PROFILE_PATH),
    output_root: str = "jobs/generated",
    dry_run: bool = False,
) -> None:
    """Generate and optionally submit an LSF job for an experiment specification."""
    spec = load_experiment_spec(config)
    lsf_profile = load_lsf_profile(profile, profiles)
    if user is not None:
        lsf_profile = _profile_for_lsf_user(lsf_profile, user)
    submission = prepare_lsf_submission(
        spec,
        lsf_profile,
        project_dir=Path.cwd(),
        output_root=output_root,
    )
    _print_submission(submission)
    if dry_run:
        return
    ctx.run(submission.bsub_command(), pty=True)


def _select_run(spec: ExperimentSpec, sweep_index: int) -> ExperimentRun:
    """Select one expanded run by one-based index."""
    runs = spec.expanded_runs()
    if sweep_index < 1 or sweep_index > len(runs):
        raise Exit(f"sweep_index must be between 1 and {len(runs)}.", code=2)
    if len(runs) > 1:
        print(f"Selected sweep run {sweep_index}/{len(runs)} from {spec.name}.")
    return runs[sweep_index - 1]


def _format_spec_summary(spec: ExperimentSpec) -> str:
    """Return a concise human-readable experiment summary."""
    parts = [
        f"Experiment: {spec.name}",
        f"Spec: {spec.path}",
        f"Command: {spec.command}",
        f"Runs: {len(spec.expanded_runs())}",
    ]
    if spec.description:
        parts.append(f"Description: {spec.description}")
    return "\n".join(parts)


def _format_sweep(run: ExperimentRun) -> str:
    """Return formatted sweep values for a concrete run."""
    if not run.sweep_values:
        return ""
    values = ", ".join(f"{key}={value}" for key, value in run.sweep_values.items())
    return f" ({values})"


def _profile_for_lsf_user(profile: LsfProfile, user: str) -> LsfProfile:
    """Return a profile with the selected user's LSF notification email."""
    normalized_user = user.strip().lower()
    try:
        email = LSF_USER_EMAILS[normalized_user]
    except KeyError as exc:
        allowed = ", ".join(sorted(LSF_USER_EMAILS))
        raise Exit(
            f"Unknown LSF user '{user}'. Use one of: {allowed}.", code=2
        ) from exc
    return replace(profile, email=email)


def _print_submission(submission: GeneratedSubmission) -> None:
    """Print generated submission paths and the bsub command."""
    print(f"Generated submission: {submission.submission_dir}")
    print(f"  script: {submission.script_path}")
    print(f"  manifest: {submission.manifest_path}")
    print(f"  run env dir: {submission.env_dir}")
    print(f"  runs: {len(submission.runs)}")
    print(f"Submit with: {submission.bsub_command()}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is below parent without requiring Python 3.9 fallback."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


namespace = Collection()
namespace.add_task(sync)
namespace.add_task(train)

experiments = Collection("experiments")
experiments.add_task(experiments_list)
experiments.add_task(experiments_plan)
namespace.add_collection(experiments)

hpc = Collection("hpc")
hpc.add_task(hpc_profiles)
hpc.add_task(hpc_storage)
hpc.add_task(hpc_submit)
namespace.add_collection(hpc)

ns = namespace
