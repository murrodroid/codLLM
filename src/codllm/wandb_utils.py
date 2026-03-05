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

    if wandb_cfg.mode in ("online", "offline"):
        os.environ["WANDB_MODE"] = wandb_cfg.mode
        return ["wandb"], wandb_cfg.run_name

    if has_wandb_credentials():
        return ["wandb"], wandb_cfg.run_name

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


def _report_to_includes_wandb(report_to: str | list[str] | None) -> bool:
    """Return True when trainer reporting includes W&B."""
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return report_to in {"wandb", "all"}
    return "wandb" in report_to


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


def _resolve_source_path(path_value: str, data_raw_dir: str) -> Path:
    """Resolve source file path against the configured raw data directory."""
    source_path = Path(path_value)
    if source_path.is_absolute() or source_path.exists():
        return source_path
    return Path(data_raw_dir) / source_path


def _build_source_metadata(cfg: Config) -> list[dict[str, Any]]:
    """Build file metadata for configured sources to aid experiment provenance."""
    source_payloads: list[dict[str, Any]] = []
    for source in cfg.data_sources:
        resolved_path = _resolve_source_path(source.path, cfg.data_raw_dir)
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
    cfg: Config, data_metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a W&B-ready metadata payload for one experiment run."""
    cfg_payload = _sanitize_for_wandb(asdict(cfg))
    if isinstance(cfg_payload, dict):
        cfg_payload.pop("hf_token", None)

    payload: dict[str, Any] = {
        "config": cfg_payload,
        "resolved": {
            "max_target_length": cfg.resolved_max_target_length(),
            "data_seed": cfg.seed if cfg.data_seed is None else cfg.data_seed,
        },
        "data_sources": _build_source_metadata(cfg),
        "runtime": _runtime_metadata(),
    }
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
    if not _report_to_includes_wandb(report_to):
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

    wandb.config.update(_sanitize_for_wandb(dict(metadata)), allow_val_change=True)
