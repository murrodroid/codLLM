from typing import Any, Callable

import numpy as np


def _normalize_decoded_text(text: str) -> str:
    """Normalize decoded text for robust exact-match comparisons."""
    return " ".join(text.strip().split())


def _split_predicted_codes(text: str, label_separator: str) -> set[str]:
    """Split one decoded prediction/label string into a normalized code set."""
    normalized = _normalize_decoded_text(text)
    if not normalized:
        return set()

    tokens = normalized.split(label_separator) if label_separator else [normalized]
    return {token.strip() for token in tokens if token.strip()}


def _precision_recall_f1(
    predictions: list[set[str]], labels: list[set[str]]
) -> dict[str, float]:
    """Compute micro precision/recall/F1 over per-example code sets."""
    tp = 0
    fp = 0
    fn = 0
    for predicted_codes, label_codes in zip(predictions, labels):
        tp += len(predicted_codes.intersection(label_codes))
        fp += len(predicted_codes.difference(label_codes))
        fn += len(label_codes.difference(predicted_codes))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float((2 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def build_exact_match_accuracy_metric(
    tokenizer: Any,
    label_separator: str = ",",
) -> Callable[[Any], dict[str, float]]:
    """Build a compute_metrics callback with exact-match and overlap metrics."""
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Compute exact-match accuracy and micro precision/recall/F1."""
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
        predicted_code_sets = [
            _split_predicted_codes(text, label_separator)
            for text in normalized_predictions
        ]
        label_code_sets = [
            _split_predicted_codes(text, label_separator) for text in normalized_labels
        ]
        overlap_metrics = _precision_recall_f1(predicted_code_sets, label_code_sets)
        return {"accuracy": accuracy, **overlap_metrics}

    return compute_metrics
