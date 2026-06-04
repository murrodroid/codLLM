from __future__ import annotations

import errno
import os
import time
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
from codllm.maintenance import (
    DatasetCacheAction,
    DatasetCacheReport,
    GitHygieneReport,
    HpcEnvironmentReport,
    MaintenanceCacheAction,
    MaintenanceStatusReport,
    build_dataset_cache_report,
    build_git_hygiene_report,
    build_hpc_environment_report,
    build_maintenance_status,
    clear_dataset_caches,
    clear_generated_caches,
    format_bytes,
    write_git_snapshot,
)

DEFAULT_EXPERIMENT_ROOT = Path("runs")
DEFAULT_PROFILE_PATH = Path("hpc/lsf_profiles.toml")
LSF_USER_EMAILS = {
    "lucas": "s234805@dtu.dk",
    "elias": "s234854@dtu.dk",
}
LSF_USER_ACCOUNTS = {
    "lucas": "s234805",
    "elias": "s234854",
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
    uv_cache = ctx.run("uv cache dir", hide=True).stdout.strip()
    _print_hpc_environment_report(
        build_hpc_environment_report(os.environ, uv_cache_dir=uv_cache)
    )


@task(name="data-cache")
def maintenance_data_cache(
    ctx: Context,
    config: str | None = None,
    sweep_index: int = 1,
    profile: str | None = None,
    profiles: str = str(DEFAULT_PROFILE_PATH),
    user: str | None = None,
    lucas: bool = False,
    elias: bool = False,
) -> None:
    """Inspect processed-data and prepared-split cache usage."""
    del ctx
    runtime_env = _maintenance_runtime_env(profile, profiles, user, lucas, elias)
    with _temporary_environ(runtime_env):
        with _config_environment(config, sweep_index):
            report = build_dataset_cache_report(config_from_env())
    _print_dataset_cache_report(report)


@task(name="clear-data-cache")
def maintenance_clear_data_cache(
    ctx: Context,
    config: str | None = None,
    sweep_index: int = 1,
    profile: str | None = None,
    profiles: str = str(DEFAULT_PROFILE_PATH),
    user: str | None = None,
    lucas: bool = False,
    elias: bool = False,
    processed: bool = True,
    splits: bool = True,
    locks: bool = False,
    temporary: bool = True,
    yes: bool = False,
) -> None:
    """Clear processed-data caches, dry-running unless --yes is provided."""
    del ctx
    runtime_env = _maintenance_runtime_env(profile, profiles, user, lucas, elias)
    with _temporary_environ(runtime_env):
        with _config_environment(config, sweep_index):
            actions = clear_dataset_caches(
                config_from_env(),
                processed=processed,
                splits=splits,
                locks=locks,
                temporary=temporary,
                execute=yes,
            )
    _print_dataset_cache_actions(actions, executed=yes)
    if not yes:
        print("Dry run only. Re-run with --yes to delete these cache paths.")


@task(name="clear-cache")
def maintenance_clear_cache(
    ctx: Context,
    config: str | None = None,
    sweep_index: int = 1,
    profile: str | None = None,
    profiles: str = str(DEFAULT_PROFILE_PATH),
    user: str | None = None,
    lucas: bool = False,
    elias: bool = False,
    standard: bool = False,
    aggressive: bool = False,
    days: int = 14,
    locks: bool = False,
    include_env_caches: bool = True,
    yes: bool = False,
) -> None:
    """Clear generated caches and outputs using standard or aggressive policy."""
    del ctx
    if standard and aggressive:
        raise Exit("Use either --standard or --aggressive, not both.", code=2)
    mode = "aggressive" if aggressive else "standard"
    runtime_env = _maintenance_runtime_env(profile, profiles, user, lucas, elias)
    with _temporary_environ(runtime_env):
        with _config_environment(config, sweep_index):
            actions = clear_generated_caches(
                config_from_env(),
                repo_dir=Path.cwd(),
                environ=os.environ,
                mode=mode,
                retention_days=days,
                locks=locks,
                include_env_caches=include_env_caches,
                execute=yes,
            )
    _print_maintenance_cache_actions(
        actions,
        executed=yes,
        mode=mode,
        retention_days=days,
    )
    if not yes:
        print("Dry run only. Re-run with --yes to delete these generated paths.")


@task(name="hpc-env")
def maintenance_hpc_env(
    ctx: Context,
    profile: str | None = None,
    profiles: str = str(DEFAULT_PROFILE_PATH),
    user: str | None = None,
    lucas: bool = False,
    elias: bool = False,
    strict: bool = False,
) -> None:
    """Inspect HPC cache/storage environment hygiene."""
    runtime_env = _maintenance_runtime_env(profile, profiles, user, lucas, elias)
    uv_cache = ctx.run("uv cache dir", env=runtime_env, hide=True).stdout.strip()
    report = build_hpc_environment_report(runtime_env, uv_cache_dir=uv_cache)
    _print_hpc_environment_report(report)
    if strict and report.has_issues:
        raise Exit("HPC environment hygiene check failed.", code=1)


@task(name="git-hygiene")
def maintenance_git_hygiene(ctx: Context, strict: bool = False) -> None:
    """Check generated logs, jobs, caches, and run outputs before pushing."""
    del ctx
    report = build_git_hygiene_report(Path.cwd())
    _print_git_hygiene_report(report)
    if strict and report.has_issues:
        raise Exit("Git hygiene check failed.", code=1)


@task(name="git-snapshot")
def maintenance_git_snapshot(
    ctx: Context,
    output_dir: str = "logs/git",
    commits: int = 20,
) -> None:
    """Write git status and recent commits to an ignored local log file."""
    del ctx
    snapshot_path = write_git_snapshot(
        Path.cwd(),
        output_dir=output_dir,
        max_commits=commits,
    )
    print(f"Wrote git snapshot: {snapshot_path}")


@task(name="status")
def maintenance_status(
    ctx: Context,
    config: str | None = None,
    sweep_index: int = 1,
    profile: str | None = None,
    profiles: str = str(DEFAULT_PROFILE_PATH),
    user: str | None = None,
    lucas: bool = False,
    elias: bool = False,
) -> None:
    """Show storage capacity and known codLLM storage usage by category."""
    del ctx
    runtime_env = _maintenance_runtime_env(profile, profiles, user, lucas, elias)
    with _temporary_environ(runtime_env):
        with _config_environment(config, sweep_index):
            report = build_maintenance_status(
                config_from_env(),
                repo_dir=Path.cwd(),
                environ=os.environ,
            )
    _print_maintenance_status(report)


@task(name="build")
def hpc_build(
    ctx: Context,
    config: str,
    profile: str = "h100-10h",
    profiles: str = str(DEFAULT_PROFILE_PATH),
    user: str | None = None,
    lucas: bool = False,
    elias: bool = False,
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
    lsf_profile = _resolve_lsf_profile(profile, profiles, user, lucas, elias)

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
    lucas: bool = False,
    elias: bool = False,
    profiles: str = str(DEFAULT_PROFILE_PATH),
    output_root: str = "jobs/generated",
    dry_run: bool = False,
) -> None:
    """Generate and optionally submit an LSF job for an experiment specification."""
    spec = load_experiment_spec(config)
    lsf_profile = _resolve_lsf_profile(profile, profiles, user, lucas, elias)
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
        try:
            splits = handler.get_splits(force_reprocess=effective_force_reprocess)
        except OSError as exc:
            _raise_hpc_storage_exit(exc, cfg.data_processed_dir)
        print(
            "  prepared splits: "
            f"train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}"
        )
        if splits.holdout is not None:
            print(f"  holdout rows: {len(splits.holdout)}")
        if splits.holdout_eval is not None:
            print(f"  holdout eval rows: {len(splits.holdout_eval)}")

        if cfg.pretrain_enabled:
            try:
                pretrain_df = handler.get_pretraining_train_dataframe()
            except OSError as exc:
                _raise_hpc_storage_exit(exc, cfg.data_processed_dir)
            pretrain_rows = 0 if pretrain_df is None else len(pretrain_df)
            print(f"  pretraining dataframe rows: {pretrain_rows}")
        if cfg.model_task == "sequence_classification":
            labels = handler.get_masterlist_label_vocabulary()
            print(f"  classifier label vocabulary: {len(labels)} labels")


def _raise_hpc_storage_exit(exc: OSError, data_processed_dir: str) -> None:
    """Raise a clearer Invoke exit for HPC quota and storage failures."""
    if exc.errno not in {errno.EDQUOT, errno.ENOSPC}:
        raise exc

    problem = (
        "disk quota exceeded"
        if exc.errno == errno.EDQUOT
        else "no space left on device"
    )
    raise Exit(
        f"HPC build could not write a dataset cache file: {problem}.\n"
        f"Processed-data cache root: {data_processed_dir}\n"
        "Check the enforced quota with:\n"
        "  uv run --no-sync invoke maintenance.status --lucas\n"
        "If no cache-build or training jobs are active, clear all managed generated data including locks with:\n"
        "  uv run --no-sync invoke maintenance.clear-cache --aggressive --locks --lucas --yes\n"
        "If quota is still full, inspect runtime dependency caches with:\n"
        "  uv run --no-sync invoke maintenance.hpc-env --lucas",
        code=1,
    ) from exc


@contextmanager
def _config_environment(config: str | None, sweep_index: int) -> Iterator[None]:
    """Apply one optional experiment run environment while resolving Config."""
    if config is None:
        yield
        return

    spec = load_experiment_spec(config)
    run = _select_run(spec, sweep_index)
    with _temporary_environ(run.env_with_runtime_metadata(spec)):
        yield


def _maintenance_runtime_env(
    profile: str | None,
    profiles: str,
    user: str | None,
    lucas: bool,
    elias: bool,
) -> dict[str, str]:
    """Return environment values for maintenance, optionally using an HPC profile."""
    selected_user = _resolve_lsf_user_alias(user, lucas, elias)
    if profile is None and selected_user is None:
        return os.environ.copy()
    profile_name = profile or "h100-10h"
    lsf_profile = _resolve_lsf_profile(profile_name, profiles, user, lucas, elias)
    env = os.environ.copy()
    env.update(_hpc_runtime_env_defaults(lsf_profile))
    if env.get("VIRTUAL_ENV") and env["VIRTUAL_ENV"] != env["UV_PROJECT_ENVIRONMENT"]:
        env.pop("VIRTUAL_ENV", None)
    return env


def _resolve_lsf_profile(
    profile: str,
    profiles: str,
    user: str | None,
    lucas: bool,
    elias: bool,
) -> LsfProfile:
    """Load an LSF profile and resolve optional user storage aliases."""
    selected_user = _resolve_lsf_user_alias(user, lucas, elias)
    lsf_profile = load_lsf_profile(profile, profiles)
    if selected_user is not None:
        return _profile_for_lsf_user(lsf_profile, selected_user)
    return _profile_with_inferred_lsf_user(lsf_profile)


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
        "HF_HOME": str(Path(run_storage_dir) / "cache/huggingface"),
        "HF_HUB_CACHE": str(Path(run_storage_dir) / "cache/huggingface/hub"),
        "TRANSFORMERS_CACHE": str(
            Path(run_storage_dir) / "cache/huggingface/transformers"
        ),
        "HF_DATASETS_CACHE": str(Path(run_storage_dir) / "cache/hf_datasets"),
        "TORCH_HOME": str(Path(run_storage_dir) / "cache/torch"),
        "WANDB_DIR": str(Path(run_storage_dir) / "cache/wandb"),
        "WANDB_CACHE_DIR": str(Path(run_storage_dir) / "cache/wandb/cache"),
        "XDG_CACHE_HOME": str(Path(run_storage_dir) / "cache/xdg"),
        "UV_CACHE_DIR": str(Path(run_storage_dir) / "cache/uv"),
        "UV_PROJECT_ENVIRONMENT": str(Path(run_storage_dir) / ".venv"),
        "UV_PYTHON_INSTALL_DIR": str(Path(run_storage_dir) / "python"),
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
        account = LSF_USER_ACCOUNTS[normalized_user]
    except KeyError as exc:
        allowed = ", ".join(sorted(LSF_USER_EMAILS))
        raise Exit(
            f"Unknown LSF user '{user}'. Use one of: {allowed}.", code=2
        ) from exc
    return replace(
        profile,
        email=email,
        storage_folder=_replace_shell_user(profile.storage_folder, account),
        run_storage_dir=_replace_shell_user(profile.run_storage_dir, account)
        if profile.run_storage_dir is not None
        else None,
    )


def _profile_with_inferred_lsf_user(profile: LsfProfile) -> LsfProfile:
    """Return a profile with storage placeholders resolved from its email when possible."""
    if profile.email is None:
        return profile
    for user, email in LSF_USER_EMAILS.items():
        if profile.email.strip().lower() == email:
            return _profile_for_lsf_user(profile, user)
    return profile


def _resolve_lsf_user_alias(
    user: str | None,
    lucas: bool,
    elias: bool,
) -> str | None:
    """Resolve --user and shortcut user flags for HPC-oriented tasks."""
    selected = [
        name for name, enabled in (("lucas", lucas), ("elias", elias)) if enabled
    ]
    if user is not None and user.strip() != "":
        selected.append(user.strip())
    if len(selected) > 1:
        raise Exit("Use only one of --user, --lucas, or --elias.", code=2)
    return selected[0] if selected else None


def _replace_shell_user(value: str, account: str) -> str:
    """Replace shell USER placeholders with one explicit HPC account."""
    return value.replace("${USER}", account).replace("$USER", account)


def _print_submission(submission: GeneratedSubmission) -> None:
    """Print generated submission paths and the bsub command."""
    print(f"Generated submission: {submission.submission_dir}")
    print(f"  script: {submission.script_path}")
    print(f"  manifest: {submission.manifest_path}")
    print(f"  run env dir: {submission.env_dir}")
    print(f"  runs: {len(submission.runs)}")
    print(f"Submit with: {submission.bsub_command()}")


def _print_dataset_cache_report(report: DatasetCacheReport) -> None:
    """Print a human-readable dataset cache report."""
    print(f"Data processed dir: {report.processed_dir}")
    for entry in report.entries:
        state = "present" if entry.exists else "missing"
        print(
            f"{entry.kind:>22}  {state:>7}  {format_bytes(entry.size_bytes):>10}  "
            f"{entry.path}"
        )
    print(f"Total existing cache size: {format_bytes(report.total_size_bytes)}")


def _print_dataset_cache_actions(
    actions: tuple[DatasetCacheAction, ...],
    *,
    executed: bool,
) -> None:
    """Print planned or executed dataset cache cleanup actions."""
    verb = "Deleted" if executed else "Would delete"
    if not actions:
        print("No matching dataset cache paths.")
        return
    for action in actions:
        if not action.exists:
            print(f"Missing {action.kind}: {action.path}")
            continue
        print(
            f"{verb} {action.kind} ({format_bytes(action.size_bytes)}): {action.path}"
        )


def _print_maintenance_cache_actions(
    actions: tuple[MaintenanceCacheAction, ...],
    *,
    executed: bool,
    mode: str,
    retention_days: int,
) -> None:
    """Print planned or executed broad cache cleanup actions."""
    if mode == "standard":
        print(
            f"Policy: standard, removing generated paths unused for {retention_days}+ days."
        )
    else:
        print("Policy: aggressive, removing all maintenance-managed generated paths.")
    if not actions:
        print("No matching generated cache paths.")
        return
    verb = "Deleted" if executed else "Would delete"
    for action in actions:
        entry = action.entry
        print(
            f"{verb} {entry.kind} ({format_bytes(entry.size_bytes)}, "
            f"age={_age_days(entry.last_activity_ns):.1f}d): {entry.path}"
        )
        print(f"  reason: {entry.reason}")


def _print_hpc_environment_report(report: HpcEnvironmentReport) -> None:
    """Print storage/cache path values and warnings."""
    for key, value in report.paths.items():
        print(f"{key}={value or '<unset>'}")
    for issue in report.issues:
        print(f"WARNING: {issue.message} {issue.value or '<unset>'}")


def _print_git_hygiene_report(report: GitHygieneReport) -> None:
    """Print generated-path git hygiene results."""
    print(f"Repository: {report.repo_dir}")
    for probe, ignored in report.ignored_probes.items():
        state = "ignored" if ignored else "not ignored"
        print(f"{state:>11}  {probe}")
    if report.status_entries:
        print("Generated/local-only git status entries:")
        for entry in report.status_entries:
            print(f"  {entry}")
    if not report.issues:
        print("Git hygiene check passed.")
        return
    print("Git hygiene warnings:")
    for issue in report.issues:
        print(f"  {issue.path}: {issue.message}")


def _print_maintenance_status(report: MaintenanceStatusReport) -> None:
    """Print storage roots and known codLLM usage categories."""
    print("Storage roots:")
    for root in report.roots:
        capacity_note = (
            f" capacity_at={root.capacity_path}"
            if root.capacity_path != root.path
            else ""
        )
        print(
            f"  {root.name}: {root.path} "
            f"filesystem_used={format_bytes(root.used_bytes)} "
            f"filesystem_free={format_bytes(root.free_bytes)} "
            f"filesystem_total={format_bytes(root.total_bytes)}"
            f"{capacity_note}"
        )
        if root.quota is None:
            continue
        if root.quota.error is not None:
            print(
                f"    quota: unavailable source={root.quota.source} "
                f"error={root.quota.error}"
            )
            continue
        if root.quota.used_bytes is None or root.quota.limit_bytes is None:
            print(f"    quota: unavailable source={root.quota.source}")
            continue
        quota_free = root.quota.free_bytes or 0
        print(
            f"    quota: used={format_bytes(root.quota.used_bytes)} "
            f"free={format_bytes(quota_free)} "
            f"limit={format_bytes(root.quota.limit_bytes)} "
            f"source={root.quota.source}"
        )
    print("Known codLLM storage:")
    for category in report.categories:
        print(
            f"  {category.name}: {format_bytes(category.size_bytes)} "
            f"({len(category.paths)} path(s))"
        )


def _age_days(last_activity_ns: int) -> float:
    """Return age in days from a nanosecond timestamp."""
    if last_activity_ns <= 0:
        return 0.0
    return max(0.0, (time.time_ns() - last_activity_ns) / 1_000_000_000 / 86400)


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

maintenance = Collection("maintenance")
maintenance.add_task(maintenance_clear_cache)
maintenance.add_task(maintenance_clear_data_cache)
maintenance.add_task(maintenance_data_cache)
maintenance.add_task(maintenance_git_hygiene)
maintenance.add_task(maintenance_git_snapshot)
maintenance.add_task(maintenance_hpc_env)
maintenance.add_task(maintenance_status)
namespace.add_collection(maintenance)

ns = namespace
