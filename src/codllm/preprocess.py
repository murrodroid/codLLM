from typing import Any, Callable, Dict, Optional

from codllm.config import Config


def build_preprocess_fn(
    cfg: Config, tokenizer: Any, max_target_length: Optional[int] = None
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a dataset preprocessing function for seq2seq training."""
    target_max_length = (
        cfg.resolved_max_target_length()
        if max_target_length is None
        else max_target_length
    )

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize source and target columns into model-ready tensors."""
        if cfg.dataset_text_column not in batch:
            raise KeyError(
                f"Missing source column '{cfg.dataset_text_column}' in batch."
            )
        if cfg.dataset_label_column not in batch:
            raise KeyError(
                f"Missing label column '{cfg.dataset_label_column}' in batch."
            )

        sources = batch[cfg.dataset_text_column]
        targets = batch[cfg.dataset_label_column]

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
        return model_inputs

    return preprocess
