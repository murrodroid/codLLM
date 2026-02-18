from typing import Any, Optional, Tuple

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from config import Config
from src.model_registry import load_base_model
from src.preprocess import build_preprocess_fn


def build_training_args(cfg: Config, has_eval: bool) -> Seq2SeqTrainingArguments:
    """Build Seq2Seq training arguments compatible with transformers v5."""
    eval_strategy = cfg.eval_strategy if has_eval else "no"
    eval_steps = cfg.eval_steps if eval_strategy == "steps" else None
    fp16 = torch.cuda.is_available() and cfg.torch_dtype in (None, "auto", "float16")
    bf16 = torch.cuda.is_available() and cfg.torch_dtype == "bfloat16"
    training_kwargs = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_steps": cfg.warmup_steps,
        "logging_steps": cfg.logging_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_strategy": cfg.save_strategy,
        "save_steps": cfg.save_steps,
        "predict_with_generate": True,
        "generation_max_length": cfg.max_target_length,
        "fp16": fp16,
        "bf16": bf16,
        "report_to": "none",
        "seed": cfg.seed,
    }
    if cfg.warmup_ratio is not None:
        training_kwargs["warmup_ratio"] = cfg.warmup_ratio
    return Seq2SeqTrainingArguments(**training_kwargs)


def train(
    cfg: Config, train_ds: Any, eval_ds: Optional[Any] = None
) -> Tuple[Seq2SeqTrainer, Any]:
    """Preprocess datasets and run a seq2seq fine-tuning job."""
    model, tokenizer = load_base_model(cfg)

    preprocess = build_preprocess_fn(cfg, tokenizer)
    train_ds = train_ds.map(
        preprocess, batched=True, remove_columns=train_ds.column_names
    )
    processed_eval_ds = None
    if eval_ds is not None:
        processed_eval_ds = eval_ds.map(
            preprocess, batched=True, remove_columns=eval_ds.column_names
        )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    args = build_training_args(cfg, has_eval=processed_eval_ds is not None)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=processed_eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    return trainer, tokenizer
