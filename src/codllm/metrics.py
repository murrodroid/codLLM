from typing import Any, Callable

import numpy as np


def _normalize_decoded_text(text: str) -> str:
    """Normalize decoded text for robust exact-match comparisons."""
    return " ".join(text.strip().split())


def build_exact_match_accuracy_metric(
    tokenizer: Any,
) -> Callable[[Any], dict[str, float]]:
    """Build a compute_metrics callback with exact-match accuracy."""
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Compute exact-match accuracy for generated predictions."""
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        prediction_ids = np.asarray(predictions)
        label_ids = np.asarray(labels)

        if prediction_ids.ndim == 3:
            prediction_ids = prediction_ids.argmax(axis=-1)

        label_ids = np.where(label_ids == -100, pad_token_id, label_ids)
        decoded_predictions = tokenizer.batch_decode(
            prediction_ids.tolist(), skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            label_ids.tolist(), skip_special_tokens=True
        )

        normalized_predictions = [
            _normalize_decoded_text(text) for text in decoded_predictions
        ]
        normalized_labels = [_normalize_decoded_text(text) for text in decoded_labels]
        matches = [
            prediction == label
            for prediction, label in zip(normalized_predictions, normalized_labels)
        ]
        accuracy = float(np.mean(matches)) if matches else 0.0
        return {"accuracy": accuracy}

    return compute_metrics
