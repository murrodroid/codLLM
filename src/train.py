import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from model_registry import load_base_model
from preprocess import build_preprocess_fn
from config import Config


def train(cfg: Config, train_ds, eval_ds):
    model, tokenizer = load_base_model(cfg)

    preprocess = build_preprocess_fn(cfg, tokenizer)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        fp16=torch.cuda.is_available() and cfg.torch_dtype in (None, "auto", "float16"),
        bf16=torch.cuda.is_available() and cfg.torch_dtype == "bfloat16",
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer, tokenizer
