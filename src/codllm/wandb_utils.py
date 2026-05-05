import json
import netrc
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import warnings

import torch

from codllm.config import Config
from codllm.runtime.paths import resolve_source_path

MAX_WANDB_ARTIFACT_METADATA_KEYS = 100
_ACTIVE_WANDB_RUN_CONFIG_MODE = "standard"
_CORE_TRANSFORMERS_SETUP_CONFIG_KEYS = frozenset(
    {
        "output_dir",
        "run_name",
        "seed",
        "data_seed",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
    }
)
_STANDARD_TRANSFORMERS_SETUP_CONFIG_KEYS = (
    _CORE_TRANSFORMERS_SETUP_CONFIG_KEYS
    | frozenset(
        {
            "lr_scheduler_type",
            "weight_decay",
            "warmup_steps",
            "max_grad_norm",
            "eval_strategy",
            "eval_steps",
            "save_strategy",
            "save_steps",
            "metric_for_best_model",
            "greater_is_better",
            "fp16",
            "bf16",
            "generation_max_length",
            "predict_with_generate",
        }
    )
)


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
    global _ACTIVE_WANDB_RUN_CONFIG_MODE
    wandb_cfg = cfg.wandb
    _ACTIVE_WANDB_RUN_CONFIG_MODE = wandb_cfg.run_config_mode

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
    cfg: Config | None = None,
) -> None:
    """Patch Transformers W&B hooks for scoped keys and compact metadata."""
    global _ACTIVE_WANDB_RUN_CONFIG_MODE
    if not report_to_includes_wandb(report_to):
        return
    if cfg is not None:
        _ACTIVE_WANDB_RUN_CONFIG_MODE = cfg.wandb.run_config_mode
    try:
        from transformers.integrations import integration_utils
    except ImportError:
        return

    current_rewrite = getattr(integration_utils, "rewrite_logs", None)
    if current_rewrite is rewrite_logs_preserving_scoped_metric_keys:
        pass
    else:
        integration_utils.rewrite_logs = rewrite_logs_preserving_scoped_metric_keys
    _patch_transformers_wandb_setup_config_filter(integration_utils)
    _patch_transformers_wandb_artifact_metadata_limit(integration_utils)


def _looks_like_transformers_setup_config(payload: Any) -> bool:
    """Return True for the broad config dict emitted by Transformers W&B setup."""
    return (
        isinstance(payload, Mapping)
        and "output_dir" in payload
        and ("learning_rate" in payload or "per_device_train_batch_size" in payload)
    )


def _filter_transformers_setup_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Trim Transformers TrainingArguments config before it reaches W&B config."""
    mode = _ACTIVE_WANDB_RUN_CONFIG_MODE
    if mode == "full":
        return _sanitize_for_wandb(dict(payload))
    allowed_keys = (
        _CORE_TRANSFORMERS_SETUP_CONFIG_KEYS
        if mode == "minimal"
        else _STANDARD_TRANSFORMERS_SETUP_CONFIG_KEYS
    )
    return {
        f"transformers.{key}": _sanitize_for_wandb(value)
        for key, value in payload.items()
        if key in allowed_keys
    }


def _patch_transformers_wandb_setup_config_filter(integration_utils: Any) -> None:
    """Patch Transformers W&B setup so it does not flood run config."""
    callback_cls = getattr(integration_utils, "WandbCallback", None)
    if callback_cls is None:
        return
    current_setup = getattr(callback_cls, "setup", None)
    if current_setup is None or getattr(current_setup, "_codllm_patched", False):
        return

    original_setup = current_setup

    def patched_setup(
        self: Any, args: Any, state: Any, model: Any, **kwargs: Any
    ) -> Any:
        wandb = getattr(self, "_wandb", None)
        patched_config_updates: dict[type[Any], Any] = {}

        def filtered_update(
            target_config: Any,
            payload: dict[str, Any],
            *update_args: Any,
            **update_kwargs: Any,
        ) -> Any:
            if _looks_like_transformers_setup_config(payload):
                payload = _filter_transformers_setup_config(payload)
            original_update = patched_config_updates[type(target_config)]
            return original_update(
                target_config,
                payload,
                *update_args,
                **update_kwargs,
            )

        def patch_config_update(config_obj: Any) -> None:
            if config_obj is None:
                return
            config_cls = type(config_obj)
            if config_cls in patched_config_updates:
                return
            original_update = getattr(config_cls, "update", None)
            if not callable(original_update):
                return
            patched_config_updates[config_cls] = original_update
            setattr(config_cls, "update", filtered_update)

        patch_config_update(getattr(wandb, "config", None))
        original_init = getattr(wandb, "init", None)

        def init_and_patch_config(*init_args: Any, **init_kwargs: Any) -> Any:
            result = original_init(*init_args, **init_kwargs)
            patch_config_update(getattr(wandb, "config", None))
            return result

        try:
            if callable(original_init):
                setattr(wandb, "init", init_and_patch_config)
        except (AttributeError, TypeError):
            original_init = None
        try:
            return original_setup(self, args, state, model, **kwargs)
        finally:
            if callable(original_init):
                try:
                    setattr(wandb, "init", original_init)
                except (AttributeError, TypeError):
                    pass
            try:
                for config_cls, original_update in patched_config_updates.items():
                    setattr(config_cls, "update", original_update)
            except (AttributeError, TypeError):
                pass

    patched_setup._codllm_patched = True
    patched_setup._codllm_original = original_setup
    callback_cls.setup = patched_setup


def _trim_wandb_artifact_metadata(
    metadata: Any,
    max_keys: int = MAX_WANDB_ARTIFACT_METADATA_KEYS,
) -> Any:
    """Limit artifact metadata keys to satisfy W&B's top-level metadata cap."""
    if not isinstance(metadata, Mapping):
        return metadata
    sanitized = _sanitize_for_wandb(dict(metadata))
    if len(sanitized) <= max_keys:
        return sanitized

    priority_keys = [
        "final_model",
        "initial_model",
        "scope",
        "stage",
        "rows",
        "model/num_parameters",
        "train/total_flos",
        "train/total_floss",
        "val/loss",
        "val/accuracy",
        "val/exact_match",
        "val/macro_f1",
        "test/accuracy",
        "test/exact_match",
        "test/macro_f1",
    ]
    trimmed: dict[str, Any] = {}
    for key in priority_keys:
        if key in sanitized and len(trimmed) < max_keys:
            trimmed[key] = sanitized[key]
    for key in sorted(sanitized):
        if key in trimmed:
            continue
        if len(trimmed) >= max_keys:
            break
        trimmed[key] = sanitized[key]
    return trimmed


def _patch_transformers_wandb_artifact_metadata_limit(integration_utils: Any) -> None:
    """Patch Transformers final model artifact logging to avoid W&B metadata limits."""
    callback_cls = getattr(integration_utils, "WandbCallback", None)
    if callback_cls is None:
        return
    current_on_train_end = getattr(callback_cls, "on_train_end", None)
    if current_on_train_end is None or getattr(
        current_on_train_end, "_codllm_patched", False
    ):
        return

    original_on_train_end = current_on_train_end

    def patched_on_train_end(
        self: Any,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        processing_class: Any = None,
        **kwargs: Any,
    ) -> Any:
        wandb = getattr(self, "_wandb", None)
        original_artifact = getattr(wandb, "Artifact", None)
        if not callable(original_artifact):
            return original_on_train_end(
                self,
                args,
                state,
                control,
                model=model,
                processing_class=processing_class,
                **kwargs,
            )

        def artifact_factory(*artifact_args: Any, **artifact_kwargs: Any) -> Any:
            if "metadata" in artifact_kwargs:
                artifact_kwargs["metadata"] = _trim_wandb_artifact_metadata(
                    artifact_kwargs["metadata"]
                )
            return original_artifact(*artifact_args, **artifact_kwargs)

        try:
            setattr(wandb, "Artifact", artifact_factory)
        except (AttributeError, TypeError):
            return original_on_train_end(
                self,
                args,
                state,
                control,
                model=model,
                processing_class=processing_class,
                **kwargs,
            )
        try:
            return original_on_train_end(
                self,
                args,
                state,
                control,
                model=model,
                processing_class=processing_class,
                **kwargs,
            )
        finally:
            try:
                setattr(wandb, "Artifact", original_artifact)
            except (AttributeError, TypeError):
                pass

    patched_on_train_end._codllm_patched = True
    patched_on_train_end._codllm_original = original_on_train_end
    callback_cls.on_train_end = patched_on_train_end


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


def _metadata_training_stage(metadata: Mapping[str, Any]) -> str:
    """Return the training-stage name embedded in a metadata payload."""
    dataset = metadata.get("dataset")
    if not isinstance(dataset, Mapping):
        return "train"
    stage = dataset.get("training_stage")
    if not isinstance(stage, Mapping):
        return "train"
    name = stage.get("name")
    return str(name).strip() or "train"


def _safe_wandb_artifact_name(value: str) -> str:
    """Normalize a value for use as a W&B artifact name."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    normalized = normalized.strip(".-")
    return normalized or "codllm-artifact"


def _minimal_wandb_config_payload(
    cfg: Config,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the smallest W&B run config needed for run-page comparisons."""
    stage = _metadata_training_stage(metadata)
    return {
        "codllm.run_config_mode": cfg.wandb.run_config_mode,
        "codllm.metric_mode": cfg.wandb.metric_mode,
        "stage.name": stage,
        "model.hf_model": cfg.hf_model,
        "model.task": cfg.model_task,
        "training.lr": cfg.lr,
        "training.epochs": cfg.num_train_epochs,
        "data.dataset_size": cfg.dataset_size,
        "data.max_label_count": cfg.max_label_count,
        "balance.strategy": cfg.balance_strategy,
        "perturbation.base_rate": cfg.base_perturbation_rate,
        "pretraining.enabled": cfg.pretrain_enabled,
    }


def _standard_wandb_config_payload(
    cfg: Config,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact but useful W&B run config for standard runs."""
    payload = _minimal_wandb_config_payload(cfg, metadata)
    payload.update(
        {
            "training.weight_decay": cfg.weight_decay,
            "training.warmup_ratio": cfg.warmup_ratio,
            "training.lr_scheduler_type": cfg.lr_scheduler_type,
            "training.batch_size.train": cfg.per_device_train_batch_size,
            "training.batch_size.eval": cfg.per_device_eval_batch_size,
            "training.gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "training.eval_strategy": cfg.eval_strategy,
            "training.save_strategy": cfg.save_strategy,
            "training.save_strategy_best_metric": cfg.save_strategy_best_metric,
            "data.train_size": cfg.train_size,
            "data.val_size": cfg.val_size,
            "data.test_size": cfg.test_size,
            "data.training_input": list(cfg.training_input),
            "data.hold_out_dataset": cfg.hold_out_dataset,
            "multicod.shuffle_labels": cfg.multicod_shuffle_labels,
            "multicod.synthetic_ratio": cfg.multicod_synthetic_ratio,
            "multicod.synthetic_source_scope": cfg.multicod_synthetic_source_scope,
            "balance.floor": cfg.balance_floor,
            "balance.floor_decay": cfg.balance_floor_decay,
            "balance.perturbations": list(cfg.balance_perturbations),
            "balance.perturbation_mean": cfg.balance_perturbation_mean,
            "balance.perturbation_variance": cfg.balance_perturbation_variance,
            "perturbation.base_perturbations": list(cfg.base_perturbations),
            "perturbation.base_mean": cfg.base_perturbation_mean,
            "perturbation.base_variance": cfg.base_perturbation_variance,
            "pretraining.epochs": cfg.pretrain_num_train_epochs,
            "pretraining.learning_rate": cfg.pretrain_learning_rate,
            "pretraining.warmup_ratio": cfg.pretrain_warmup_ratio,
            "pretraining.multicod_synthetic_ratio": cfg.pretrain_multicod_synthetic_ratio,
            "seed.global": cfg.seed,
            "seed.data": cfg.data_seed,
            "seed.resolved_data": cfg.resolved_data_seed(),
        }
    )
    runtime = metadata.get("runtime")
    if isinstance(runtime, Mapping):
        payload["runtime.git_commit"] = runtime.get("git_commit")
        hpc_env = runtime.get("hpc_env")
        if isinstance(hpc_env, Mapping):
            for key in (
                "WANDB_SWEEP_ID",
                "WANDB_RUN_GROUP",
                "LSB_JOBID",
                "LSB_JOBINDEX",
            ):
                if key in hpc_env:
                    payload[f"runtime.{key.lower()}"] = hpc_env[key]
    dataset = metadata.get("dataset")
    if isinstance(dataset, Mapping):
        split_rows = dataset.get("split_rows")
        if isinstance(split_rows, Mapping):
            for split_name, row_count in split_rows.items():
                payload[f"dataset.split_rows.{split_name}"] = row_count
        classification = dataset.get("classification")
        if isinstance(classification, Mapping):
            payload["classification.num_labels"] = classification.get("num_labels")
    return _sanitize_for_wandb(payload)


def _wandb_config_payload(
    cfg: Config,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the selected run-page config payload for W&B."""
    sanitized_metadata = _sanitize_for_wandb(dict(metadata))
    if cfg.wandb.run_config_mode == "full":
        return sanitized_metadata if isinstance(sanitized_metadata, dict) else {}
    if cfg.wandb.run_config_mode == "minimal":
        return _sanitize_for_wandb(_minimal_wandb_config_payload(cfg, metadata))
    return _standard_wandb_config_payload(cfg, metadata)


def _log_wandb_metadata_artifact(
    wandb: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Log the full experiment metadata as a JSON artifact."""
    run = getattr(wandb, "run", None)
    log_artifact = getattr(run, "log_artifact", None)
    artifact_factory = getattr(wandb, "Artifact", None)
    if not callable(log_artifact) or not callable(artifact_factory):
        return

    stage = _metadata_training_stage(metadata)
    run_id = str(getattr(run, "id", "") or getattr(run, "name", "") or "run")
    artifact_name = _safe_wandb_artifact_name(
        f"{run_id}-codllm-experiment-metadata-{stage}"
    )
    config_payload = metadata.get("config")
    raw_wandb_payload = (
        config_payload.get("wandb") if isinstance(config_payload, Mapping) else None
    )
    wandb_payload = raw_wandb_payload if isinstance(raw_wandb_payload, Mapping) else {}
    artifact_metadata = _trim_wandb_artifact_metadata(
        {
            "stage": stage,
            "run_config_mode": wandb_payload.get("run_config_mode"),
            "metric_mode": wandb_payload.get("metric_mode"),
        }
    )
    artifact = artifact_factory(
        name=artifact_name,
        type="run_metadata",
        metadata=artifact_metadata,
    )
    filename = f"experiment_metadata_{_safe_wandb_artifact_name(stage)}.json"
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = Path(temp_dir) / filename
        metadata_path.write_text(
            json.dumps(_sanitize_for_wandb(dict(metadata)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact.add_file(str(metadata_path), name=filename)
        log_artifact(artifact, aliases=[_safe_wandb_artifact_name(stage), "latest"])


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

    config_payload = _wandb_config_payload(cfg, metadata)
    if config_payload:
        wandb.config.update(config_payload, allow_val_change=True)
    if cfg.wandb.run_config_mode == "full" and isinstance(config_payload, Mapping):
        flattened_metadata = _flatten_mapping_for_wandb(config_payload)
        if flattened_metadata:
            wandb.config.update(flattened_metadata, allow_val_change=True)
    _log_wandb_metadata_artifact(wandb, metadata)
