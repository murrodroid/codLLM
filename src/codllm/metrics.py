from typing import Any, Callable, Mapping

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


def _per_class_stats(
    predictions: list[set[str]], labels: list[set[str]]
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Accumulate per-class TP, label totals, and prediction totals."""
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
    return class_tp, class_label_total, class_pred_total


def _macro_from_class_stats(
    class_tp: dict[str, int],
    class_label_total: dict[str, int],
    class_pred_total: dict[str, int],
    classes: set[str] | None = None,
) -> dict[str, float]:
    """Compute macro P/R/F1 over the given class subset (or all classes)."""
    all_classes = classes if classes is not None else (set(class_label_total) | set(class_pred_total))
    if not all_classes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

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
        "precision": float(np.mean(per_class_precision)),
        "recall": float(np.mean(per_class_recall)),
        "f1": float(np.mean(per_class_f1)),
    }


def _sanitize_token_ids_for_decoding(
    token_ids: np.ndarray,
    *,
    pad_token_id: int,
    max_token_id: int | None = None,
) -> np.ndarray:
    """Return token ids safe for tokenizer.decode/batch_decode.

    Tokenizer decoders can fail on invalid integral values (for example negative
    ids, NaN/Inf values cast from floating tensors, or very large out-of-range ids).
    This helper maps those invalid ids to the configured pad token id.
    """
    sanitized = np.asarray(token_ids)
    if not np.issubdtype(sanitized.dtype, np.integer):
        finite_mask = np.isfinite(sanitized)
        sanitized = np.where(finite_mask, sanitized, pad_token_id)
        sanitized = np.rint(sanitized).astype(np.int64, copy=False)
    else:
        sanitized = sanitized.astype(np.int64, copy=False)

    sanitized = np.where(sanitized < 0, pad_token_id, sanitized)
    if max_token_id is not None:
        sanitized = np.where(sanitized > max_token_id, pad_token_id, sanitized)
    return sanitized
def _macro_precision_recall_f1(
    predictions: list[set[str]], labels: list[set[str]]
) -> dict[str, float]:
    """Compute macro-averaged precision/recall/F1 across all classes."""
    class_tp, class_label_total, class_pred_total = _per_class_stats(predictions, labels)
    raw = _macro_from_class_stats(class_tp, class_label_total, class_pred_total)
    return {f"macro_{k}": v for k, v in raw.items()}


def _seen_unseen_macro(
    predictions: list[set[str]],
    labels: list[set[str]],
    train_classes: set[str],
) -> dict[str, float]:
    """Compute macro P/R/F1 split by seen (in training) and unseen classes.

    - seen_macro_*: metrics over eval classes that appeared in training
    - unseen_macro_*: metrics over eval classes never seen during training
    - seen_class_count / unseen_class_count: how many classes in each bucket
    """
    class_tp, class_label_total, class_pred_total = _per_class_stats(predictions, labels)
    eval_classes = set(class_label_total) | set(class_pred_total)

    seen = eval_classes & train_classes
    unseen = eval_classes - train_classes

    result: dict[str, float] = {
        "seen_class_count": float(len(seen)),
        "unseen_class_count": float(len(unseen)),
    }

    seen_stats = _macro_from_class_stats(class_tp, class_label_total, class_pred_total, seen)
    for k, v in seen_stats.items():
        result[f"seen_macro_{k}"] = v

    unseen_stats = _macro_from_class_stats(class_tp, class_label_total, class_pred_total, unseen)
    for k, v in unseen_stats.items():
        result[f"unseen_macro_{k}"] = v

    return result


def build_exact_match_accuracy_metric(
    tokenizer: Any,
    label_separator: str = ",",
    max_label_count: int = 1,
    train_classes: set[str] | None = None,
) -> Callable[[Any], dict[str, float]]:
    """Build a compute_metrics callback with exact-match and overlap metrics.

    When max_label_count is 1 (single-label), micro metrics are skipped because
    they are mathematically identical to accuracy in that regime.

    When train_classes is provided, additional seen/unseen macro metrics are
    emitted to distinguish performance on classes the model trained on vs.
    classes it has never seen.
    """
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    raw_vocab_size = getattr(tokenizer, "vocab_size", None)
    max_token_id = (
        int(raw_vocab_size) - 1
        if isinstance(raw_vocab_size, int) and raw_vocab_size > 0
        else None
    )
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

        prediction_ids = _sanitize_token_ids_for_decoding(
            prediction_ids,
            pad_token_id=pad_token_id,
            max_token_id=max_token_id,
        )
        label_ids = np.where(label_ids == -100, pad_token_id, label_ids)
        label_ids = _sanitize_token_ids_for_decoding(
            label_ids,
            pad_token_id=pad_token_id,
            max_token_id=max_token_id,
        )
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
            result.update(
                _micro_precision_recall_f1(predicted_code_sets, label_code_sets)
            )
        result.update(_macro_precision_recall_f1(predicted_code_sets, label_code_sets))
        return result

    return compute_metrics


def build_sequence_classification_metric(
    id2label: Mapping[int, str],
) -> Callable[[Any], dict[str, float]]:
    """Build compute_metrics callback for single-label sequence classification."""
    normalized_id2label = {int(key): str(value) for key, value in id2label.items()}

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Compute accuracy and macro precision/recall/F1 from classifier logits."""
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        prediction_array = np.asarray(predictions)
        if prediction_array.ndim == 2:
            predicted_ids = prediction_array.argmax(axis=-1)
        else:
            predicted_ids = prediction_array

        label_ids = np.asarray(labels)
        predicted_ids = predicted_ids.astype(np.int64, copy=False)
        label_ids = label_ids.astype(np.int64, copy=False)

        normalized_predictions = [
            normalized_id2label.get(int(label_id), str(int(label_id)))
            for label_id in predicted_ids.tolist()
        ]
        normalized_labels = [
            normalized_id2label.get(int(label_id), str(int(label_id)))
            for label_id in label_ids.tolist()
        ]

        matches = [
            prediction == label
            for prediction, label in zip(normalized_predictions, normalized_labels)
        ]
        accuracy = float(np.mean(matches)) if matches else 0.0

        predicted_code_sets = [{label} for label in normalized_predictions]
        label_code_sets = [{label} for label in normalized_labels]
        result: dict[str, float] = {"accuracy": accuracy}
        result.update(_macro_precision_recall_f1(predicted_code_sets, label_code_sets))
        if train_classes is not None:
            result.update(_seen_unseen_macro(predicted_code_sets, label_code_sets, train_classes))
        return result

    return compute_metrics
