import argparse
from typing import Any, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from codllm.config import Config
from codllm.config import config as default_config
from codllm.data_handler import DataHandler, DataSplits
from codllm.model_registry import load_base_model
from codllm.preprocess import build_preprocess_fn


class TokenizedSeq2SeqDataset(Dataset):
    """Simple torch dataset wrapper for tokenized seq2seq features."""

    def __init__(self, features: dict[str, list[Any]]) -> None:
        if not features:
            raise ValueError("Tokenized features must not be empty.")
        lengths = {len(values) for values in features.values()}
        if len(lengths) != 1:
            raise ValueError("Tokenized feature lengths are inconsistent.")
        self.features = features
        self.length = lengths.pop()

    def __len__(self) -> int:
        """Return number of rows."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one tokenized training sample."""
        return {key: values[idx] for key, values in self.features.items()}


def _tokenize_dataframe(
    cfg: Config, tokenizer: Any, dataframe: pd.DataFrame
) -> TokenizedSeq2SeqDataset:
    """Convert a pandas dataframe into a tokenized seq2seq dataset."""
    if dataframe.empty:
        raise ValueError("Training dataframe is empty.")
    if cfg.dataset_text_column not in dataframe.columns:
        raise KeyError(
            f"Missing source column '{cfg.dataset_text_column}' in dataframe."
        )
    if cfg.dataset_label_column not in dataframe.columns:
        raise KeyError(
            f"Missing label column '{cfg.dataset_label_column}' in dataframe."
        )

    sources = dataframe[cfg.dataset_text_column].fillna("").astype(str).tolist()
    targets = dataframe[cfg.dataset_label_column].fillna("").astype(str).tolist()

    model_inputs = tokenizer(
        sources,
        max_length=cfg.max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=cfg.max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return TokenizedSeq2SeqDataset(model_inputs)


def _prepare_dataset(cfg: Config, tokenizer: Any, dataset: Any) -> Any:
    """Convert an input dataset into the tokenized format expected by Trainer."""
    if isinstance(dataset, pd.DataFrame):
        return _tokenize_dataframe(cfg, tokenizer, dataset)

    if hasattr(dataset, "map") and hasattr(dataset, "column_names"):
        preprocess = build_preprocess_fn(cfg, tokenizer)
        return dataset.map(
            preprocess, batched=True, remove_columns=dataset.column_names
        )

    raise TypeError(
        "Unsupported dataset type. Expected pandas.DataFrame or a dataset with map/column_names."
    )


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

    processed_train_ds = _prepare_dataset(cfg, tokenizer, train_ds)
    processed_eval_ds = None
    if eval_ds is not None:
        processed_eval_ds = _prepare_dataset(cfg, tokenizer, eval_ds)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    args = build_training_args(cfg, has_eval=processed_eval_ds is not None)

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


def train_with_data_handler(
    cfg: Config,
    data_handler: Optional[DataHandler] = None,
    force_reprocess: bool = False,
) -> Tuple[Seq2SeqTrainer, Any, DataSplits]:
    """Build/load data splits via DataHandler and launch training."""

    handler = data_handler or DataHandler(cfg)
    splits = handler.get_splits(force_reprocess=force_reprocess)
    eval_dataset: Optional[pd.DataFrame]
    if splits.val.empty:
        eval_dataset = None
    else:
        eval_dataset = splits.val
    trainer, tokenizer = train(cfg, splits.train, eval_dataset)
    return trainer, tokenizer, splits


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for running fine-tuning from the command line."""
    parser = argparse.ArgumentParser(description="Run seq2seq fine-tuning.")
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Rebuild processed data even when a processed file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch fine-tuning using the default project config."""
    args = _parse_args()
    _, _, splits = train_with_data_handler(
        cfg=default_config,
        force_reprocess=args.force_reprocess,
    )
    print(
        f"Training completed. train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}."
    )


if __name__ == "__main__":
    main()
