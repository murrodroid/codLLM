from typing import Any, Mapping

import pandas as pd
from torch.utils.data import Dataset

from codllm.config import Config
from codllm.data.preprocess import build_preprocess_fn


class TokenizedSeq2SeqDataset(Dataset):
    """Simple torch dataset wrapper for tokenized seq2seq features."""

    def __init__(
        self,
        features: dict[str, list[Any]],
        source_ids: list[str] | None = None,
    ) -> None:
        if not features:
            raise ValueError("Tokenized features must not be empty.")
        lengths = {len(values) for values in features.values()}
        if len(lengths) != 1:
            raise ValueError("Tokenized feature lengths are inconsistent.")
        self.features = features
        self.length = lengths.pop()
        if source_ids is not None and len(source_ids) != self.length:
            raise ValueError("source_ids length must match tokenized feature length.")
        self.source_ids: list[str] | None = (
            list(source_ids) if source_ids is not None else None
        )

    def __len__(self) -> int:
        """Return number of rows."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one tokenized training sample."""
        return {key: values[idx] for key, values in self.features.items()}


def _tokenize_dataframe(
    cfg: Config,
    tokenizer: Any,
    dataframe: pd.DataFrame,
    target_max_length: int,
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
        max_length=target_max_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    source_ids: list[str] | None = None
    if "source_id" in dataframe.columns:
        source_ids = dataframe["source_id"].fillna("").astype(str).tolist()
    return TokenizedSeq2SeqDataset(model_inputs, source_ids=source_ids)


def prepare_training_dataset(
    cfg: Config,
    tokenizer: Any,
    dataset: Any,
    target_max_length: int,
) -> Any:
    """Convert a dataset into the tokenized format expected by Seq2SeqTrainer."""
    if isinstance(dataset, pd.DataFrame):
        return _tokenize_dataframe(cfg, tokenizer, dataset, target_max_length)

    if hasattr(dataset, "map") and hasattr(dataset, "column_names"):
        preprocess = build_preprocess_fn(
            cfg,
            tokenizer,
            max_target_length=target_max_length,
        )
        return dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
        )

    raise TypeError(
        "Unsupported dataset type. Expected pandas.DataFrame or a dataset with map/column_names."
    )


def _normalize_label_value(value: Any) -> str:
    """Normalize one label value to a stripped string."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _tokenize_dataframe_for_sequence_classification(
    cfg: Config,
    tokenizer: Any,
    dataframe: pd.DataFrame,
    label2id: Mapping[str, int],
) -> TokenizedSeq2SeqDataset:
    """Convert a pandas dataframe into a tokenized sequence-classification dataset."""
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
    labels = [
        _normalize_label_value(value) for value in dataframe[cfg.dataset_label_column]
    ]

    unique_labels = set(labels)
    unknown_labels = sorted(label for label in unique_labels if label not in label2id)
    if unknown_labels:
        preview = ", ".join(unknown_labels[:10])
        raise ValueError(
            f"Found labels missing from classifier label space: {preview}."
        )

    model_inputs = tokenizer(
        sources,
        max_length=cfg.max_source_length,
        truncation=True,
    )
    model_inputs["labels"] = [int(label2id[label]) for label in labels]
    source_ids: list[str] | None = None
    if "source_id" in dataframe.columns:
        source_ids = dataframe["source_id"].fillna("").astype(str).tolist()
    return TokenizedSeq2SeqDataset(model_inputs, source_ids=source_ids)


def prepare_sequence_classification_dataset(
    cfg: Config,
    tokenizer: Any,
    dataset: Any,
    label2id: Mapping[str, int],
) -> Any:
    """Convert a dataset into tokenized format expected by Trainer classification."""
    if isinstance(dataset, pd.DataFrame):
        return _tokenize_dataframe_for_sequence_classification(
            cfg=cfg,
            tokenizer=tokenizer,
            dataframe=dataset,
            label2id=label2id,
        )

    if hasattr(dataset, "map") and hasattr(dataset, "column_names"):

        def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
            if cfg.dataset_text_column not in batch:
                raise KeyError(
                    f"Missing source column '{cfg.dataset_text_column}' in batch."
                )
            if cfg.dataset_label_column not in batch:
                raise KeyError(
                    f"Missing label column '{cfg.dataset_label_column}' in batch."
                )
            sources = batch[cfg.dataset_text_column]
            raw_labels = batch[cfg.dataset_label_column]
            normalized_labels = [_normalize_label_value(value) for value in raw_labels]
            unknown_labels = sorted(
                label for label in set(normalized_labels) if label not in label2id
            )
            if unknown_labels:
                preview = ", ".join(unknown_labels[:10])
                raise ValueError(
                    f"Found labels missing from classifier label space: {preview}."
                )
            model_inputs = tokenizer(
                sources,
                max_length=cfg.max_source_length,
                truncation=True,
            )
            model_inputs["labels"] = [
                int(label2id[label]) for label in normalized_labels
            ]
            return model_inputs

        return dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
        )

    raise TypeError(
        "Unsupported dataset type. Expected pandas.DataFrame or a dataset with map/column_names."
    )
