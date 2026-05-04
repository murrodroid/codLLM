import netrc
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import warnings

import torch

from codllm.config import Config
from codllm.runtime.paths import resolve_source_path


def has_wandb_credentials() -> bool:
    """Return True when W&B credentials are available via env or netrc."""
    if os.getenv("WANDB_API_KEY"):
        return True

    for filename in (".netrc", "_netrc"):
        netrc_path = Path.home() / filename
        if not netrc_path.exists():
            continue
        try:
            auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
        except (OSError, netrc.NetrcParseError):
            continue
        if auth and auth[2]:
            return True
    return False


def _auto_run_name(cfg: Config) -> str:
    """Build a descriptive W&B run name from the config."""
    model_short = cfg.hf_model.split("/")[-1]
    parts = [model_short]
    if cfg.balance_strategy == "none":
        parts.append("no-upsample")
    elif cfg.balance_strategy == "floor":
        parts.append(f"floor{cfg.balance_floor}")
        if cfg.balance_floor_decay > 0:
            parts.append(f"decay{cfg.balance_floor_decay}")
    parts.append(f"{cfg.num_train_epochs}ep")
    inputs = ",".join(cfg.training_input)
    parts.append(inputs)
    return "_".join(parts)


def resolve_wandb_reporting(cfg: Config) -> tuple[str | list[str], str | None]:
    """Resolve Trainer reporting settings and runtime environment for W&B."""
    wandb_cfg = cfg.wandb

    if not wandb_cfg.enabled or wandb_cfg.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_LOG_MODEL"] = "false"
        return "none", None

    os.environ.setdefault("WANDB_PROJECT", wandb_cfg.project)
    if wandb_cfg.entity:
        os.environ["WANDB_ENTITY"] = wandb_cfg.entity
    os.environ["WANDB_LOG_MODEL"] = wandb_cfg.log_model

    run_name = wandb_cfg.run_name or _auto_run_name(cfg)

    if wandb_cfg.mode in ("online", "offline"):
        os.environ["WANDB_MODE"] = wandb_cfg.mode
        return ["wandb"], run_name

    if has_wandb_credentials():
        return ["wandb"], run_name

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_LOG_MODEL"] = "false"
    warnings.warn(
        (
            "W&B is enabled but no WANDB_API_KEY or ~/.netrc credentials were found. "
            "Falling back to report_to='none'."
        ),
        stacklevel=2,
    )
    return "none", None


def report_to_includes_wandb(report_to: str | list[str] | None) -> bool:
    """Return True when trainer reporting includes W&B."""
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return report_to in {"wandb", "all"}
    return "wandb" in report_to


def rewrite_logs_preserving_scoped_metric_keys(
    logs: Mapping[str, Any],
) -> dict[str, Any]:
    """Rewrite W&B logs while preserving already-scoped metric keys."""
    rewritten_logs: dict[str, Any] = {}
    for key, value in logs.items():
        if key in {"epoch", "step", "global_step"}:
            rewritten_logs[key] = value
            continue
        if "/" in key:
            rewritten_logs[key] = value
            continue
        if key.startswith("eval_"):
            rewritten_logs[f"val/{key.removeprefix('eval_')}"] = value
            continue
        if key.startswith("test_"):
            rewritten_logs[f"test/{key.removeprefix('test_')}"] = value
            continue
        if key.startswith("holdout_val_"):
            rewritten_logs[f"holdout/val/{key.removeprefix('holdout_val_')}"] = value
            continue
        if key.startswith("holdout_test_"):
            rewritten_logs[f"holdout/test/{key.removeprefix('holdout_test_')}"] = value
            continue
        if key.startswith("holdout_"):
            rewritten_logs[f"holdout/{key.removeprefix('holdout_')}"] = value
            continue
        rewritten_logs[f"train/{key}"] = value
    return rewritten_logs


def patch_transformers_wandb_log_rewrite(
    report_to: str | list[str] | None,
) -> None:
    """Patch Transformers W&B rewrite hook to preserve scoped metric keys."""
    if not report_to_includes_wandb(report_to):
        return
    try:
        from transformers.integrations import integration_utils
    except ImportError:
        return

    current_rewrite = getattr(integration_utils, "rewrite_logs", None)
    if current_rewrite is rewrite_logs_preserving_scoped_metric_keys:
        return
    integration_utils.rewrite_logs = rewrite_logs_preserving_scoped_metric_keys


def _sanitize_for_wandb(value: Any) -> Any:
    """Convert nested Python objects into W&B-config-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_for_wandb(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_wandb(item) for item in value]
    return str(value)


def _flatten_mapping_for_wandb(
    payload: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Flatten nested mapping keys into dot notation for W&B config views."""
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping_for_wandb(value, prefix=dotted_key))
            continue
        flattened[dotted_key] = _sanitize_for_wandb(value)
    return flattened


def _build_source_metadata(cfg: Config) -> list[dict[str, Any]]:
    """Build file metadata for configured sources to aid experiment provenance."""
    source_payloads: list[dict[str, Any]] = []
    for source in cfg.data_sources:
        resolved_path = resolve_source_path(source.path, cfg.data_raw_dir)
        payload = _sanitize_for_wandb(asdict(source))
        payload["resolved_path"] = str(resolved_path.resolve())
        payload["exists"] = resolved_path.exists()
        if resolved_path.exists():
            stats = resolved_path.stat()
            payload["size_bytes"] = stats.st_size
            payload["mtime_ns"] = stats.st_mtime_ns
        source_payloads.append(payload)
    return source_payloads


def _safe_git_commit() -> str | None:
    """Return current git commit hash when available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return commit or None


def _runtime_metadata() -> dict[str, Any]:
    """Collect runtime metadata useful for reproducibility and HPC debugging."""
    hpc_env_keys = [
        "LSB_JOBID",
        "LSB_JOBINDEX",
        "LSB_JOBNAME",
        "LSB_QUEUE",
        "LSB_HOSTS",
        "CUDA_VISIBLE_DEVICES",
        "WANDB_SWEEP_ID",
        "WANDB_RUN_GROUP",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "UV_PROJECT_ENVIRONMENT",
    ]
    hpc_env = {key: value for key in hpc_env_keys if (value := os.getenv(key))}

    cuda_device_names: list[str] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            cuda_device_names.append(torch.cuda.get_device_name(idx))

    transformers_version: str | None = None
    try:
        import transformers
    except ImportError:
        transformers_version = None
    else:
        transformers_version = transformers.__version__

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "cuda_device_names": cuda_device_names,
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": _safe_git_commit(),
        "hpc_env": hpc_env,
    }


def build_experiment_metadata(
    cfg: Config,
    data_metadata: Mapping[str, Any] | None = None,
    training_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a W&B-ready metadata payload for one experiment run."""
    cfg_payload = _sanitize_for_wandb(asdict(cfg))
    if isinstance(cfg_payload, dict):
        cfg_payload.pop("hf_token", None)

    payload: dict[str, Any] = {
        "config": cfg_payload,
        "resolved": {
            "max_target_length": cfg.resolved_max_target_length(),
            "data_seed": cfg.resolved_data_seed(),
        },
        "data_sources": _build_source_metadata(cfg),
        "runtime": _runtime_metadata(),
    }
    if training_args is not None:
        payload["training_args"] = _sanitize_for_wandb(dict(training_args))
    if data_metadata is not None:
        payload["dataset"] = _sanitize_for_wandb(dict(data_metadata))
    return payload


def _import_wandb() -> Any:
    """Import and return the wandb module."""
    import wandb

    return wandb


def log_wandb_run_metadata(
    cfg: Config,
    report_to: str | list[str] | None,
    run_name: str | None,
    metadata: Mapping[str, Any],
) -> None:
    """Initialize W&B if needed and attach experiment metadata to run config."""
    if not report_to_includes_wandb(report_to):
        return
    if os.getenv("WANDB_MODE") == "disabled":
        return

    try:
        wandb = _import_wandb()
    except ImportError:
        warnings.warn(
            "W&B reporting was requested but wandb is not installed.",
            stacklevel=2,
        )
        return

    if getattr(wandb, "run", None) is None:
        init_kwargs: dict[str, Any] = {
            "project": os.getenv("WANDB_PROJECT", cfg.wandb.project),
            "entity": cfg.wandb.entity or os.getenv("WANDB_ENTITY"),
            "name": run_name or cfg.wandb.run_name,
            "job_type": "train",
        }
        mode = os.getenv("WANDB_MODE")
        if mode in {"online", "offline", "disabled"}:
            init_kwargs["mode"] = mode
        wandb.init(
            **{key: value for key, value in init_kwargs.items() if value is not None}
        )

    if getattr(wandb, "run", None) is None:
        return

    # Define epoch as a step metric so eval metrics can be plotted against it
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")
    wandb.define_metric("holdout/val/*", step_metric="epoch")
    wandb.define_metric("holdout/test/*", step_metric="epoch")
    wandb.define_metric("pretraining/*", step_metric="epoch")
    wandb.define_metric("pretraining/val/*", step_metric="epoch")
    wandb.define_metric("pretraining/test/*", step_metric="epoch")
    # Pin key metrics to summary for easy comparison across runs
    for prefix in ("val", "test", "holdout/val", "holdout/test"):
        for metric_name in [
            "accuracy",
            "exact_match",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "micro_jaccard",
            "sample_f1",
            "sample_jaccard",
            "hamming_score",
            "seen_macro_f1",
            "unseen_macro_f1",
            "seen_accuracy",
            "unseen_accuracy",
        ]:
            wandb.define_metric(f"{prefix}/{metric_name}", summary="max")
        for metric_name in [
            "hamming_loss",
            "label_count_mae",
            "seen_hamming_loss",
            "unseen_hamming_loss",
        ]:
            wandb.define_metric(f"{prefix}/{metric_name}", summary="min")

    sanitized_metadata = _sanitize_for_wandb(dict(metadata))
    wandb.config.update(sanitized_metadata, allow_val_change=True)
    if isinstance(sanitized_metadata, Mapping):
        flattened_metadata = _flatten_mapping_for_wandb(sanitized_metadata)
        if flattened_metadata:
            wandb.config.update(flattened_metadata, allow_val_change=True)
