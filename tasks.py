from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from invoke import Collection, Context, Exit, task

from codllm.config import config_from_env
from codllm.data import DataHandler
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


@task(name="build")
def hpc_build(
    ctx: Context,
    config: str,
    profile: str = "h100-10h",
    profiles: str = str(DEFAULT_PROFILE_PATH),
    sweep_index: int = 0,
    force_reprocess: bool | None = None,
    dry_run: bool = False,
) -> None:
    """Build reusable data caches for an experiment spec without starting training."""
    del ctx
    spec = load_experiment_spec(config)
    if spec.command != "train":
        raise Exit(
            f"Task 'hpc.build' only supports command='train' specs, got {spec.command!r}.",
            code=2,
        )
    lsf_profile = load_lsf_profile(profile, profiles)

    runs = _select_runs(spec, sweep_index)
    print(
        f"Building data dependencies for {len(runs)} run(s) from {spec.path} "
        f"with profile {lsf_profile.name}."
    )
    for run_number, run in enumerate(runs, start=1):
        _build_run_dependencies(
            spec=spec,
            run=run,
            profile=lsf_profile,
            run_number=run_number,
            total_runs=len(runs),
            force_reprocess=force_reprocess,
            dry_run=dry_run,
        )


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


def _select_runs(spec: ExperimentSpec, sweep_index: int) -> list[ExperimentRun]:
    """Select one or all expanded runs for build-oriented tasks."""
    if sweep_index == 0:
        return spec.expanded_runs()
    return [_select_run(spec, sweep_index)]


def _build_run_dependencies(
    spec: ExperimentSpec,
    run: ExperimentRun,
    profile: LsfProfile,
    run_number: int,
    total_runs: int,
    force_reprocess: bool | None = None,
    dry_run: bool = False,
) -> None:
    """Build cached data artifacts for one expanded experiment run."""
    if run.command != "train":
        raise Exit(
            f"Task 'hpc.build' only supports command='train' runs, got {run.command!r}.",
            code=2,
        )

    effective_force_reprocess = (
        run.force_reprocess if force_reprocess is None else force_reprocess
    )
    print(f"[{run_number}/{total_runs}] {run.name}")
    if dry_run:
        print(
            "  would build processed data and prepared splits "
            f"(force_reprocess={int(effective_force_reprocess)})"
        )
        return

    runtime_defaults = _hpc_runtime_env_defaults(profile)
    run_env = runtime_defaults | run.env_with_runtime_metadata(spec)
    with _temporary_environ(run_env):
        cfg = config_from_env()
        print(f"  CODLLM_DATA_RAW_DIR={cfg.data_raw_dir}")
        print(f"  CODLLM_DATA_PROCESSED_DIR={cfg.data_processed_dir}")
        print(f"  CODLLM_OUTPUT_DIR={cfg.output_dir}")
        handler = DataHandler(cfg)
        splits = handler.get_splits(force_reprocess=effective_force_reprocess)
        print(
            "  prepared splits: "
            f"train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}"
        )
        if splits.holdout is not None:
            print(f"  holdout rows: {len(splits.holdout)}")
        if splits.holdout_eval is not None:
            print(f"  holdout eval rows: {len(splits.holdout_eval)}")

        if cfg.pretrain_enabled:
            pretrain_df = handler.get_pretraining_train_dataframe()
            pretrain_rows = 0 if pretrain_df is None else len(pretrain_df)
            print(f"  pretraining dataframe rows: {pretrain_rows}")
        if cfg.model_task == "sequence_classification":
            labels = handler.get_masterlist_label_vocabulary()
            print(f"  classifier label vocabulary: {len(labels)} labels")


def _hpc_runtime_env_defaults(profile: LsfProfile) -> dict[str, str]:
    """Return login-node defaults that mirror the generated LSF script."""
    storage_folder = os.environ.get("STORAGE_FOLDER") or os.path.expandvars(
        profile.storage_folder
    )
    if profile.run_storage_dir:
        profile_run_storage_dir = os.path.expandvars(profile.run_storage_dir)
    else:
        profile_run_storage_dir = str(Path(storage_folder) / "codllm")
    run_storage_dir = os.environ.get("RUN_STORAGE_DIR") or profile_run_storage_dir
    project_dir = Path.cwd()

    defaults = {
        "STORAGE_FOLDER": storage_folder,
        "RUN_STORAGE_DIR": run_storage_dir,
        "CODLLM_OUTPUT_DIR": str(Path(run_storage_dir) / "runs"),
        "CODLLM_DATA_RAW_DIR": str(project_dir / "data/raw"),
        "CODLLM_DATA_PROCESSED_DIR": str(Path(run_storage_dir) / "data/processed"),
    }
    return {key: os.environ.get(key, value) for key, value in defaults.items()}


@contextmanager
def _temporary_environ(updates: dict[str, str]) -> Iterator[None]:
    """Temporarily apply environment overrides while preserving the caller env."""
    previous_values = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


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
hpc.add_task(hpc_build)
hpc.add_task(hpc_profiles)
hpc.add_task(hpc_storage)
hpc.add_task(hpc_submit)
namespace.add_collection(hpc)

ns = namespace
