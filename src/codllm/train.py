import argparse
from typing import Any, Optional, Tuple
import warnings

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
from codllm.model_registry import load_base_model
from codllm.reproducibility import configure_reproducibility


def build_training_args(
    cfg: Config,
    has_eval: bool,
    generation_max_length: Optional[int] = None,
    disable_fp16: bool = False,
) -> Seq2SeqTrainingArguments:
    """Build Seq2Seq training arguments compatible with transformers v5."""
    eval_strategy = cfg.eval_strategy if has_eval else "no"
    eval_steps = cfg.eval_steps if eval_strategy == "steps" else None
    using_cuda = torch.cuda.is_available()
    bf16_supported = (
        using_cuda
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
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
    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    resolved_generation_max_length = (
        cfg.resolved_max_target_length()
        if generation_max_length is None
        else generation_max_length
    )
    training_kwargs = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": cfg.num_train_epochs,
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


def _train_from_datasets(
    cfg: Config, train_ds: Any, eval_ds: Optional[Any] = None
) -> Tuple[Seq2SeqTrainer, Any]:
    """Preprocess datasets and run a seq2seq fine-tuning job."""
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
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=processed_train_ds,
        eval_dataset=processed_eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    return trainer, tokenizer


def train(
    cfg: Config,
    data_handler: Optional[DataHandler] = None,
    force_reprocess: bool = False,
) -> Tuple[Seq2SeqTrainer, Any, DataSplits]:
    """Build/load data splits via DataHandler and launch training."""

    handler = data_handler or DataHandler(cfg)
    splits = handler.get_splits(force_reprocess=force_reprocess)
    train_ds, eval_ds = resolve_training_frames(splits)
    trainer, tokenizer = _train_from_datasets(cfg, train_ds, eval_ds)
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
