from typing import Any
import warnings

from transformers import Seq2SeqTrainingArguments, TrainingArguments

import codllm.wandb_utils as wandb_utils
from codllm.config import Config
from codllm.training.stages import TrainingStage, build_train_stage


def _metric_greater_is_better(metric_name: str) -> bool:
    """Return whether higher metric values indicate better checkpoints."""
    normalized_metric = metric_name.strip().lower()
    return not normalized_metric.endswith("loss")


def _metric_requires_multi_label(metric_name: str) -> bool:
    """Return whether a best-checkpoint metric is emitted only for multi-label runs."""
    normalized_metric = metric_name.strip().lower()
    return (
        normalized_metric.startswith("micro_")
        or normalized_metric.startswith("sample_")
        or normalized_metric in {"hamming_loss", "hamming_score"}
    )


def build_training_args(
    cfg: Config,
    has_eval: bool,
    stage: TrainingStage | None = None,
    generation_max_length: int | None = None,
    disable_fp16: bool = False,
    include_inputs_for_metrics: bool | None = None,
) -> TrainingArguments:
    """Build training arguments for seq2seq or classification stages."""
    resolved_stage = stage or build_train_stage(cfg)
    eval_strategy = cfg.eval_strategy if has_eval else "no"
    eval_steps = cfg.eval_steps if eval_strategy == "steps" else None
    metric_inputs_enabled = (
        has_eval if include_inputs_for_metrics is None else include_inputs_for_metrics
    )
    if cfg.save_strategy == "best" and eval_strategy == "no":
        raise ValueError(
            "save_strategy='best' requires validation data and eval_strategy "
            "set to 'steps' or 'epoch'."
        )
    if (
        cfg.save_strategy == "best"
        and _metric_requires_multi_label(cfg.save_strategy_best_metric)
        and cfg.max_label_count <= 1
    ):
        raise ValueError(
            "The configured save_strategy_best_metric is only emitted when "
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
    resolved_generation_max_length = (
        cfg.resolved_max_target_length()
        if generation_max_length is None
        else generation_max_length
    )
    dataloader_num_workers = cfg.dataloader_num_workers
    training_kwargs: dict[str, Any] = {
        "output_dir": resolved_stage.output_dir,
        "learning_rate": resolved_stage.learning_rate,
        "lr_scheduler_type": resolved_stage.lr_scheduler_type,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": resolved_stage.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_steps": cfg.logging_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "include_for_metrics": ["inputs"] if metric_inputs_enabled else [],
        "save_strategy": cfg.save_strategy,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "fp16": fp16,
        "bf16": bf16,
        "report_to": report_to,
        "run_name": run_name,
        "seed": cfg.seed,
        "data_seed": cfg.resolved_data_seed(),
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": cfg.dataloader_pin_memory,
        # Avoid HF's O(completed-steps) dataloader replay on resume, which
        # otherwise stalls every resubmitted slot with the GPU idle.
        "ignore_data_skip": cfg.ignore_data_skip,
    }
    if dataloader_num_workers > 0:
        training_kwargs["dataloader_persistent_workers"] = (
            cfg.dataloader_persistent_workers
        )
        training_kwargs["dataloader_prefetch_factor"] = cfg.dataloader_prefetch_factor
    else:
        training_kwargs["dataloader_persistent_workers"] = False
    if cfg.save_strategy == "best":
        training_kwargs["metric_for_best_model"] = cfg.save_strategy_best_metric
        training_kwargs["greater_is_better"] = _metric_greater_is_better(
            cfg.save_strategy_best_metric
        )

    if cfg.load_best_model_at_end:
        if eval_strategy == "no":
            raise ValueError(
                "load_best_model_at_end=True requires validation data and "
                "eval_strategy set to 'steps' or 'epoch'."
            )
        if (
            _metric_requires_multi_label(cfg.save_strategy_best_metric)
            and cfg.max_label_count <= 1
        ):
            raise ValueError(
                "load_best_model_at_end=True with save_strategy_best_metric "
                f"'{cfg.save_strategy_best_metric}' requires max_label_count > 1; "
                "this metric is only emitted for multi-label runs."
            )
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs.setdefault(
            "metric_for_best_model", cfg.save_strategy_best_metric
        )
        training_kwargs.setdefault(
            "greater_is_better",
            _metric_greater_is_better(cfg.save_strategy_best_metric),
        )
    if resolved_stage.warmup_ratio >= 0:
        training_kwargs["warmup_steps"] = resolved_stage.warmup_ratio
    if cfg.model_task == "seq2seq":
        training_kwargs["predict_with_generate"] = True
        training_kwargs["generation_max_length"] = resolved_generation_max_length
        return Seq2SeqTrainingArguments(**training_kwargs)
    return TrainingArguments(**training_kwargs)
