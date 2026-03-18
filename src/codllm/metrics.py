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


def _micro_precision_recall_f1(
    predictions: list[set[str]], labels: list[set[str]]
) -> dict[str, float]:
    """Compute micro-averaged precision/recall/F1 over per-example code sets."""
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
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def _macro_precision_recall_f1(
    predictions: list[set[str]], labels: list[set[str]]
) -> dict[str, float]:
    """Compute macro-averaged precision/recall/F1 across all classes.

    Each unique code in the label or prediction sets is treated as a class.
    Per-class metrics are computed independently, then averaged with equal
    weight to rare and frequent classes.
    """
    class_tp: dict[str, int] = {}
    class_label_total: dict[str, int] = {}
    class_pred_total: dict[str, int] = {}
    for predicted_codes, label_codes in zip(predictions, labels):
        for code in label_codes:
            class_label_total[code] = class_label_total.get(code, 0) + 1
            if code in predicted_codes:
                class_tp[code] = class_tp.get(code, 0) + 1
        for code in predicted_codes:
            class_pred_total[code] = class_pred_total.get(code, 0) + 1

    all_classes = set(class_label_total) | set(class_pred_total)
    if not all_classes:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    for code in all_classes:
        tp = class_tp.get(code, 0)
        pred_total = class_pred_total.get(code, 0)
        label_total = class_label_total.get(code, 0)

        p = tp / pred_total if pred_total > 0 else 0.0
        r = tp / label_total if label_total > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        per_class_precision.append(p)
        per_class_recall.append(r)
        per_class_f1.append(f)

    return {
        "macro_precision": float(np.mean(per_class_precision)),
        "macro_recall": float(np.mean(per_class_recall)),
        "macro_f1": float(np.mean(per_class_f1)),
    }


def build_exact_match_accuracy_metric(
    tokenizer: Any,
    label_separator: str = ",",
    max_label_count: int = 1,
) -> Callable[[Any], dict[str, float]]:
    """Build a compute_metrics callback with exact-match and overlap metrics.

    When max_label_count is 1 (single-label), micro metrics are skipped because
    they are mathematically identical to accuracy in that regime.
    """
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    multi_label = max_label_count > 1

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Compute exact-match accuracy and class-level metrics."""
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

        result: dict[str, float] = {"accuracy": accuracy}
        if multi_label:
            result.update(_micro_precision_recall_f1(predicted_code_sets, label_code_sets))
        result.update(_macro_precision_recall_f1(predicted_code_sets, label_code_sets))
        return result

    return compute_metrics
