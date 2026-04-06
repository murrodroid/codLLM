import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Tuple
import warnings

from filelock import FileLock, Timeout
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import codllm.wandb_utils as wandb_utils
from codllm.config import Config, config_from_env
from codllm.data_handler import (
    DataHandler,
    DataSplits,
    prepare_training_dataset,
    resolve_training_frames,
)
from codllm.metrics import build_exact_match_accuracy_metric
from codllm.model_registry import load_base_model
from codllm.reproducibility import configure_reproducibility

LOCAL_RUN_DIR_PATTERN = re.compile(r"^run-(\d+)$")
DEFAULT_RUN_DIR_LOCK_TIMEOUT_SECONDS = 120.0


def _run_dir_lock_timeout_seconds() -> float:
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


def _metric_greater_is_better(metric_name: str) -> bool:
    """Return whether higher metric values indicate better checkpoints."""
    normalized_metric = metric_name.strip().lower()
    return not normalized_metric.endswith("loss")


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


def _resolve_run_output_dir(base_output_dir: str) -> Path:
    """Resolve one run-scoped output directory below the configured root."""
    output_root = Path(base_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run-dir.lock"
    lock_timeout_seconds = _run_dir_lock_timeout_seconds()
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


def _prepare_run_output_dir(cfg: Config) -> Path:
    """Mutate config output_dir to a run-scoped checkpoint root."""
    run_dir = _resolve_run_output_dir(cfg.output_dir)
    cfg.output_dir = str(run_dir)
    os.environ["CODLLM_RUN_ID"] = run_dir.name.removeprefix("run-")
    return run_dir


def _dataset_row_count(dataset: Any) -> int | None:
    """Return dataset row count when available."""
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except (TypeError, ValueError):
        return None


def _source_distribution(dataset: Any) -> dict[str, int] | None:
    """Return per-source counts when the split contains source_id metadata."""
    if dataset is None or not hasattr(dataset, "columns"):
        return None
    if "source_id" not in dataset.columns:
        return None
    counts = dataset["source_id"].fillna("unknown").astype(str).value_counts()
    return {source: int(count) for source, count in counts.items()}


def _label_stats(dataset: Any, label_column: str) -> dict[str, int] | None:
    """Return non-empty and unique label counts for a split."""
    if dataset is None or not hasattr(dataset, "columns"):
        return None
    if label_column not in dataset.columns:
        return None
    labels = dataset[label_column].fillna("").astype(str).str.strip()
    non_empty = labels[labels != ""]
    return {
        "non_empty_count": int(non_empty.shape[0]),
        "unique_count": int(non_empty.nunique()),
    }


def _load_json_file(path: Path) -> dict[str, Any] | None:
    """Load JSON file contents when present and parseable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _build_data_metadata(
    cfg: Config, splits: DataSplits, force_reprocess: bool, handler: DataHandler
) -> dict[str, Any]:
    """Build split and processed-data metadata for W&B run config."""
    source_distribution = {
        "train": _source_distribution(splits.train),
        "val": _source_distribution(splits.val),
        "test": _source_distribution(splits.test),
    }
    label_stats = {
        "train": _label_stats(splits.train, cfg.dataset_label_column),
        "val": _label_stats(splits.val, cfg.dataset_label_column),
        "test": _label_stats(splits.test, cfg.dataset_label_column),
    }

    default_processed_path = Path(cfg.data_processed_dir) / cfg.processed_filename
    processed_path = Path(getattr(handler, "processed_path", default_processed_path))
    default_metadata_path = processed_path.with_suffix(
        f"{processed_path.suffix}.meta.json"
    )
    processed_metadata_path = Path(
        getattr(handler, "processed_metadata_path", default_metadata_path)
    )

    payload: dict[str, Any] = {
        "force_reprocess": force_reprocess,
        "processed_path": str(processed_path.resolve()),
        "processed_metadata_path": str(processed_metadata_path.resolve()),
        "split_rows": {
            "train": int(len(splits.train)),
            "val": int(len(splits.val)),
            "test": int(len(splits.test)),
        },
        "split_source_distribution": {
            split: counts
            for split, counts in source_distribution.items()
            if counts is not None
        },
        "split_label_stats": {
            split: stats for split, stats in label_stats.items() if stats is not None
        },
    }
    fingerprint = _load_json_file(processed_metadata_path)
    if fingerprint is not None:
        payload["processed_fingerprint"] = fingerprint
    return payload


def _serialize_for_terminal(value: Any) -> Any:
    """Convert nested values into JSON-serializable terminal output."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize_for_terminal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_for_terminal(item) for item in value]
    return str(value)


def _print_training_configuration(
    cfg: Config,
    args: Seq2SeqTrainingArguments,
    run_data_metadata: Optional[dict[str, Any]],
) -> None:
    """Print resolved run configuration and hyperparameters when verbosity is enabled."""
    if not cfg.verbose:
        return

    cfg_payload = _serialize_for_terminal(asdict(cfg))
    if isinstance(cfg_payload, dict):
        cfg_payload.pop("hf_token", None)

    training_args_payload: dict[str, Any] = {}
    if hasattr(args, "to_dict"):
        training_args_payload = _serialize_for_terminal(args.to_dict())

    payload: dict[str, Any] = {
        "run_id": os.getenv("CODLLM_RUN_ID"),
        "output_dir": cfg.output_dir,
        "config": cfg_payload,
        "training_args": training_args_payload,
    }
    if run_data_metadata is not None:
        payload["dataset"] = _serialize_for_terminal(run_data_metadata)

    print("Resolved training setup:")
    print(json.dumps(payload, sort_keys=True, indent=2))


def build_training_args(
    cfg: Config,
    has_eval: bool,
    generation_max_length: Optional[int] = None,
    disable_fp16: bool = False,
    output_dir: Optional[str] = None,
    num_train_epochs: Optional[int] = None,
) -> Seq2SeqTrainingArguments:
    """Build Seq2Seq training arguments compatible with transformers v5."""
    eval_strategy = cfg.eval_strategy if has_eval else "no"
    eval_steps = cfg.eval_steps if eval_strategy == "steps" else None
    if cfg.save_strategy == "best" and eval_strategy == "no":
        raise ValueError(
            "save_strategy='best' requires validation data and eval_strategy "
            "set to 'steps' or 'epoch'."
        )
    if (
        cfg.save_strategy == "best"
        and cfg.save_strategy_best_metric.startswith("micro_")
        and cfg.max_label_count <= 1
    ):
        raise ValueError(
            "save_strategy_best_metric with a 'micro_' prefix requires "
            "max_label_count > 1."
        )
    using_cuda = cfg.uses_cuda()
    bf16_supported = cfg.bf16_amp_supported()
    requested_dtype = cfg.torch_dtype
    fp16 = using_cuda and requested_dtype == "float16" and not disable_fp16
    bf16 = using_cuda and (
        requested_dtype == "bfloat16" or (requested_dtype == "auto" and bf16_supported)
    )
    if requested_dtype == "bfloat16" and using_cuda and not bf16_supported:
        warnings.warn(
            "torch_dtype='bfloat16' requested, but current CUDA hardware does not support bf16 AMP.",
            stacklevel=2,
        )
    if requested_dtype == "auto" and using_cuda and not bf16_supported:
        warnings.warn(
            (
                "torch_dtype='auto' on CUDA now defaults to full precision unless bf16 is supported. "
                "Set torch_dtype='float16' explicitly to force fp16 AMP."
            ),
            stacklevel=2,
        )
    report_to, run_name = wandb_utils.resolve_wandb_reporting(cfg)
    data_seed = cfg.resolved_data_seed()
    resolved_generation_max_length = (
        cfg.resolved_max_target_length()
        if generation_max_length is None
        else generation_max_length
    )
    training_kwargs = {
        "output_dir": cfg.output_dir if output_dir is None else output_dir,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": (
            cfg.num_train_epochs if num_train_epochs is None else num_train_epochs
        ),
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_grad_norm": cfg.max_grad_norm,
        "warmup_steps": cfg.warmup_steps,
        "logging_steps": cfg.logging_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_strategy": cfg.save_strategy,
        "save_steps": cfg.save_steps,
        "predict_with_generate": True,
        "generation_max_length": resolved_generation_max_length,
        "fp16": fp16,
        "bf16": bf16,
        "report_to": report_to,
        "run_name": run_name,
        "seed": cfg.seed,
        "data_seed": data_seed,
        "dataloader_num_workers": cfg.dataloader_num_workers,
    }
    if cfg.save_strategy == "best":
        training_kwargs["metric_for_best_model"] = cfg.save_strategy_best_metric
        training_kwargs["greater_is_better"] = _metric_greater_is_better(
            cfg.save_strategy_best_metric
        )
    if cfg.warmup_steps is not None:
        training_kwargs["warmup_steps"] = cfg.warmup_steps
    return Seq2SeqTrainingArguments(**training_kwargs)


def _validate_trainable_model(model: Any) -> None:
    """Ensure current pipeline is not asked to full-finetune a quantized base model."""
    if getattr(model, "is_quantized", False):
        raise ValueError(
            "Quantized model detected for fine-tuning. "
            "Set CODLLM_LOAD_IN_8BIT=0 (or cfg.load_in_8bit=False) "
            "or attach PEFT adapters before training."
        )


def _model_uses_trainable_fp16_params(model: Any) -> bool:
    """Return True when any trainable floating-point parameter is already float16."""
    if not hasattr(model, "parameters"):
        return False
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if torch.is_floating_point(parameter) and parameter.dtype == torch.float16:
            return True
    return False


def _upcast_trainable_fp16_params(model: Any) -> bool:
    """Cast model to float32 when trainable parameters are float16."""
    if not _model_uses_trainable_fp16_params(model):
        return False
    if not hasattr(model, "float"):
        return False
    model.float()
    return True


def _initialize_training_components(cfg: Config) -> tuple[Any, Any, bool]:
    """Load model/tokenizer once and apply training-safety dtype guards."""
    configure_reproducibility(cfg)
    model, tokenizer = load_base_model(cfg)
    _validate_trainable_model(model)
    upcasted_fp16_model = _upcast_trainable_fp16_params(model)
    disable_fp16 = _model_uses_trainable_fp16_params(model)
    if upcasted_fp16_model:
        warnings.warn(
            (
                "Trainable model parameters were loaded as float16. "
                "Upcasting model to float32 to improve optimization stability."
            ),
            stacklevel=2,
        )
    if disable_fp16:
        warnings.warn(
            (
                "Trainable model parameters are already float16. "
                "Disabling Trainer fp16 AMP to avoid grad unscale errors."
            ),
            stacklevel=2,
        )
    return model, tokenizer, disable_fp16


def _with_training_stage_metadata(
    run_data_metadata: Optional[dict[str, Any]],
    stage_name: str,
    train_ds: Any,
    eval_ds: Optional[Any],
    num_train_epochs: int,
) -> dict[str, Any]:
    """Attach stage-level training metadata to the run payload."""
    stage_metadata = dict(run_data_metadata) if run_data_metadata is not None else {}
    stage_metadata["training_stage"] = {
        "name": stage_name,
        "num_train_epochs": num_train_epochs,
        "rows": {
            "train": _dataset_row_count(train_ds),
            "eval": _dataset_row_count(eval_ds),
        },
    }
    return stage_metadata


def _train_with_model(
    cfg: Config,
    model: Any,
    tokenizer: Any,
    disable_fp16: bool,
    train_ds: Any,
    eval_ds: Optional[Any] = None,
    run_data_metadata: Optional[dict[str, Any]] = None,
) -> Seq2SeqTrainer:
    """Preprocess datasets and run one seq2seq training stage."""
    stage_name = "train"
    stage_output_dir = cfg.output_dir
    stage_num_train_epochs = cfg.num_train_epochs
    if isinstance(run_data_metadata, dict):
        stage = run_data_metadata.get("training_stage")
        if isinstance(stage, dict):
            raw_stage_name = stage.get("name")
            if isinstance(raw_stage_name, str) and raw_stage_name.strip():
                stage_name = raw_stage_name.strip()
            raw_stage_epochs = stage.get("num_train_epochs")
            if isinstance(raw_stage_epochs, int):
                stage_num_train_epochs = raw_stage_epochs
            raw_stage_output_dir = stage.get("output_dir")
            if isinstance(raw_stage_output_dir, str) and raw_stage_output_dir.strip():
                stage_output_dir = raw_stage_output_dir.strip()

    target_max_length = cfg.resolved_max_target_length()
    processed_train_ds = prepare_training_dataset(
        cfg, tokenizer, train_ds, target_max_length
    )
    processed_eval_ds = None
    if eval_ds is not None:
        processed_eval_ds = prepare_training_dataset(
            cfg, tokenizer, eval_ds, target_max_length
        )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    args = build_training_args(
        cfg,
        has_eval=processed_eval_ds is not None,
        generation_max_length=target_max_length,
        disable_fp16=disable_fp16,
        output_dir=stage_output_dir,
        num_train_epochs=stage_num_train_epochs,
    )
    fallback_data_metadata = {
        "split_rows": {
            "train": _dataset_row_count(train_ds),
            "eval": _dataset_row_count(eval_ds),
        },
        "training_stage": {
            "name": stage_name,
            "num_train_epochs": stage_num_train_epochs,
            "output_dir": stage_output_dir,
        },
    }
    training_args_payload = args.to_dict() if hasattr(args, "to_dict") else None
    metadata_payload = wandb_utils.build_experiment_metadata(
        cfg=cfg,
        data_metadata=(
            run_data_metadata
            if run_data_metadata is not None
            else fallback_data_metadata
        ),
        training_args=training_args_payload,
    )
    wandb_utils.log_wandb_run_metadata(
        cfg=cfg,
        report_to=args.report_to,
        run_name=args.run_name,
        metadata=metadata_payload,
    )
    _print_training_configuration(cfg, args, run_data_metadata)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=processed_train_ds,
        eval_dataset=processed_eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=(
            build_exact_match_accuracy_metric(
                tokenizer,
                label_separator=cfg.label_separator,
                max_label_count=cfg.max_label_count,
            )
            if processed_eval_ds is not None
            else None
        ),
    )

    trainer.train()
    return trainer


def _train_from_datasets(
    cfg: Config,
    train_ds: Any,
    eval_ds: Optional[Any] = None,
    run_data_metadata: Optional[dict[str, Any]] = None,
) -> Tuple[Seq2SeqTrainer, Any]:
    """Preprocess datasets and run a seq2seq fine-tuning job."""
    model, tokenizer, disable_fp16 = _initialize_training_components(cfg)
    trainer = _train_with_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        disable_fp16=disable_fp16,
        train_ds=train_ds,
        eval_ds=eval_ds,
        run_data_metadata=run_data_metadata,
    )
    return trainer, tokenizer


def _train_with_pretraining(
    cfg: Config,
    pretrain_ds: Any,
    train_ds: Any,
    eval_ds: Optional[Any] = None,
    run_data_metadata: Optional[dict[str, Any]] = None,
) -> Tuple[Seq2SeqTrainer, Any]:
    """Run optional pretraining first, then continue with regular fine-tuning."""
    model, tokenizer, disable_fp16 = _initialize_training_components(cfg)
    pretrain_output_dir = str(Path(cfg.output_dir) / "pretrain")
    pretrain_metadata = _with_training_stage_metadata(
        run_data_metadata=run_data_metadata,
        stage_name="pretrain",
        train_ds=pretrain_ds,
        eval_ds=eval_ds,
        num_train_epochs=cfg.pretrain_num_train_epochs,
    )
    pretrain_metadata["training_stage"]["output_dir"] = pretrain_output_dir
    _ = _train_with_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        disable_fp16=disable_fp16,
        train_ds=pretrain_ds,
        eval_ds=eval_ds,
        run_data_metadata=pretrain_metadata,
    )

    finetune_output_dir = str(Path(cfg.output_dir) / "finetune")
    finetune_metadata = _with_training_stage_metadata(
        run_data_metadata=run_data_metadata,
        stage_name="finetune",
        train_ds=train_ds,
        eval_ds=eval_ds,
        num_train_epochs=cfg.num_train_epochs,
    )
    finetune_metadata["training_stage"]["output_dir"] = finetune_output_dir
    trainer = _train_with_model(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        disable_fp16=disable_fp16,
        train_ds=train_ds,
        eval_ds=eval_ds,
        run_data_metadata=finetune_metadata,
    )
    return trainer, tokenizer


def _evaluate_test_split(
    cfg: Config, trainer: Any, tokenizer: Any, test_ds: Any
) -> dict[str, float] | None:
    """Run final evaluation on the test split and emit test-prefixed metrics."""
    if _dataset_row_count(test_ds) in (None, 0):
        return None
    if not hasattr(trainer, "evaluate"):
        return None

    target_max_length = cfg.resolved_max_target_length()
    processed_test_ds = prepare_training_dataset(
        cfg, tokenizer, test_ds, target_max_length
    )
    raw_metrics = trainer.evaluate(
        eval_dataset=processed_test_ds,
        metric_key_prefix="test",
    )
    return {key: float(value) for key, value in raw_metrics.items()}


def train(
    cfg: Config,
    data_handler: Optional[DataHandler] = None,
    force_reprocess: bool = False,
) -> Tuple[Seq2SeqTrainer, Any, DataSplits]:
    """Build/load data splits via DataHandler and launch training."""
    _prepare_run_output_dir(cfg)
    handler = data_handler or DataHandler(cfg)
    splits = handler.get_splits(force_reprocess=force_reprocess)
    train_ds, eval_ds = resolve_training_frames(splits)
    run_data_metadata = _build_data_metadata(
        cfg=cfg,
        splits=splits,
        force_reprocess=force_reprocess,
        handler=handler,
    )
    pretrain_loader = getattr(handler, "get_pretraining_train_dataframe", None)
    pretrain_ds = pretrain_loader() if callable(pretrain_loader) else None
    if cfg.pretrain_enabled and pretrain_loader is None:
        raise AttributeError(
            "Configured data_handler does not support pretraining datasets."
        )
    if cfg.pretrain_enabled and pretrain_ds is None:
        raise ValueError(
            "Pretraining is enabled but no pretraining dataset was returned."
        )
    if cfg.pretrain_enabled and pretrain_ds is not None:
        if eval_ds is None:
            raise ValueError(
                "Pretraining requires a non-empty validation split from the regular dataset."
            )
        run_data_metadata["pretraining"] = {
            "enabled": True,
            "masterlist_path": str(Path(cfg.pretrain_masterlist_path).resolve()),
            "sheet_name": cfg.pretrain_masterlist_sheet_name,
            "train_rows": int(len(pretrain_ds)),
            "num_train_epochs": cfg.pretrain_num_train_epochs,
            "apply_balance": cfg.pretrain_apply_balance,
        }
        trainer, tokenizer = _train_with_pretraining(
            cfg=cfg,
            pretrain_ds=pretrain_ds,
            train_ds=train_ds,
            eval_ds=eval_ds,
            run_data_metadata=run_data_metadata,
        )
    else:
        trainer, tokenizer = _train_from_datasets(
            cfg,
            train_ds,
            eval_ds,
            run_data_metadata=run_data_metadata,
        )
    _evaluate_test_split(
        cfg=cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        test_ds=splits.test,
    )
    return trainer, tokenizer, splits


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for running fine-tuning from the command line."""
    parser = argparse.ArgumentParser(description="Run seq2seq fine-tuning.")
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Rebuild processed data even when a processed file already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed for reproducible runs.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Override config data seed for split/sampler reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch fine-tuning using the default project config."""
    args = _parse_args()
    cfg = config_from_env()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_seed is not None:
        cfg.data_seed = args.data_seed
    _, _, splits = train(
        cfg=cfg,
        force_reprocess=args.force_reprocess,
    )
    print(
        f"Training completed. train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}."
    )


if __name__ == "__main__":
    main()
