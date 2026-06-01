from collections.abc import Mapping as MappingABC
from contextvars import ContextVar, Token
from typing import Any, Callable, Mapping, Sequence

import numpy as np

MetricArtifactLogger = Callable[..., None]


_METRIC_ARTIFACT_SCOPE: ContextVar[str | None] = ContextVar(
    "codllm_metric_artifact_scope",
    default=None,
)
_METRIC_SOURCE_IDS_SCOPE: ContextVar[list[str] | None] = ContextVar(
    "codllm_metric_source_ids",
    default=None,
)
_PER_SOURCE_METRIC_PREFIX = "source_"
_HIERARCHY_METRIC_PREFIXES: tuple[str, ...] = ("chapter_", "block_")
_PER_SOURCE_AGGREGATE_EXCLUDED = frozenset({"historic_strings_en_2024"})
_CORE_LOGGED_METRICS = frozenset(
    {
        "accuracy",
        "exact_match",
        "macro_f1",
        "sample_f1",
        "sample_jaccard",
        "micro_jaccard",
        "hamming_loss",
        "chapter_block_accuracy",
        "chapter_block_macro_f1",
        "source_transfer_label_accuracy",
        "source_transfer_label_recall",
    }
)
_STANDARD_LOGGED_METRICS = _CORE_LOGGED_METRICS | frozenset(
    {
        "macro_precision",
        "macro_recall",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "hamming_score",
        "label_count_mae",
        "chapter_block_exact_match",
        "chapter_block_macro_precision",
        "chapter_block_macro_recall",
        "chapter_block_micro_precision",
        "chapter_block_micro_recall",
        "chapter_block_micro_f1",
        "chapter_block_micro_jaccard",
        "chapter_block_sample_precision",
        "chapter_block_sample_recall",
        "chapter_block_sample_f1",
        "chapter_block_sample_jaccard",
        "source_transfer_label_sample_count",
        "source_transfer_label_sample_rate",
        "source_transfer_label_true_count",
        "source_transfer_label_macro_f1",
        "source_transfer_label_chapter_block_accuracy",
        "source_transfer_label_chapter_block_macro_f1",
        "same_source_label_sample_count",
        "same_source_label_sample_rate",
        "same_source_label_true_count",
        "same_source_label_accuracy",
        "same_source_label_recall",
        "same_source_label_macro_f1",
        "same_source_label_chapter_block_accuracy",
        "same_source_label_chapter_block_macro_f1",
        "unseen_label_sample_count",
        "unseen_label_sample_rate",
        "unseen_label_true_count",
        "unseen_label_accuracy",
        "unseen_label_recall",
        "unseen_label_macro_f1",
        "unseen_label_chapter_block_accuracy",
        "unseen_label_chapter_block_macro_f1",
    }
)


def set_metric_artifact_scope(scope: str | None) -> Token[str | None]:
    """Set the current metric artifact scope for side-channel visualization logs."""
    return _METRIC_ARTIFACT_SCOPE.set(scope)


def reset_metric_artifact_scope(token: Token[str | None]) -> None:
    """Reset the metric artifact scope to a previous context value."""
    _METRIC_ARTIFACT_SCOPE.reset(token)


def current_metric_artifact_scope() -> str | None:
    """Return the current metric artifact scope."""
    return _METRIC_ARTIFACT_SCOPE.get()


def set_metric_source_ids(
    source_ids: list[str] | None,
) -> Token[list[str] | None]:
    """Set source ids parallel to the rows of the dataset currently being evaluated."""
    return _METRIC_SOURCE_IDS_SCOPE.set(source_ids)


def reset_metric_source_ids(token: Token[list[str] | None]) -> None:
    """Reset the metric source ids to a previous context value."""
    _METRIC_SOURCE_IDS_SCOPE.reset(token)


def current_metric_source_ids() -> list[str] | None:
    """Return the source ids registered for the current evaluation, if any."""
    return _METRIC_SOURCE_IDS_SCOPE.get()


def _sanitize_source_key(source_id: str) -> str:
    """Make a source_id safe to embed in a metric key."""
    cleaned = []
    for character in source_id:
        if character.isalnum() or character in {"_", "-"}:
            cleaned.append(character)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown"


def filter_metrics_for_logging(
    metrics: Mapping[str, float],
    *,
    mode: str,
    save_metric: str | None = None,
) -> dict[str, float]:
    """Return the scalar metrics selected for Trainer and W&B run-page logging."""
    normalized_mode = mode.strip().lower()
    if normalized_mode == "all":
        return dict(metrics)
    if normalized_mode == "core":
        allowed = set(_CORE_LOGGED_METRICS)
    elif normalized_mode == "standard":
        allowed = set(_STANDARD_LOGGED_METRICS)
    else:
        raise ValueError("metric logging mode must be one of: all, core, standard.")
    if save_metric:
        allowed.add(save_metric.strip().lower())
    extra_prefixes = (_PER_SOURCE_METRIC_PREFIX, *_HIERARCHY_METRIC_PREFIXES)
    return {
        key: value
        for key, value in metrics.items()
        if key in allowed or key.startswith(extra_prefixes)
    }


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


def _split_label_value(value: Any, label_separator: str) -> set[str]:
    """Split one dataset label value into a normalized code set."""
    if isinstance(value, list | tuple | set):
        return {str(item).strip() for item in value if str(item).strip()}
    return _split_predicted_codes(str(value), label_separator)


def collect_label_classes(
    dataset: Any,
    label_column: str,
    label_separator: str,
) -> set[str]:
    """Collect normalized label classes from a raw training dataset."""
    label_values = None
    if hasattr(dataset, "columns") and label_column in dataset.columns:
        label_values = dataset[label_column].tolist()
    elif isinstance(dataset, MappingABC) and label_column in dataset:
        label_values = dataset[label_column]
    elif hasattr(dataset, "column_names") and label_column in dataset.column_names:
        label_values = dataset[label_column]

    if label_values is None:
        return set()

    classes: set[str] = set()
    for value in label_values:
        classes.update(_split_label_value(value, label_separator))
    return classes


def _tokenizer_max_token_id(tokenizer: Any) -> int | None:
    """Return the highest valid tokenizer id when the tokenizer exposes a vocab size."""
    raw_vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(raw_vocab_size, int) and raw_vocab_size > 0:
        return int(raw_vocab_size) - 1
    return None


def _tokenizer_pad_token_id(tokenizer: Any) -> int:
    """Return a safe pad token id for tokenizer decoding."""
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


def _coerce_token_id_batches(token_ids: Any) -> list[list[int]]:
    """Coerce a token-id column or tensor into a list of token-id batches."""
    if token_ids is None:
        return []
    if isinstance(token_ids, tuple):
        if not token_ids:
            return []
        token_ids = token_ids[0]
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().numpy()
    if isinstance(token_ids, list | tuple):
        if not token_ids:
            return []
        first = token_ids[0]
        if isinstance(first, list | tuple | np.ndarray) or hasattr(first, "detach"):
            batches: list[list[int]] = []
            for sequence in token_ids:
                if hasattr(sequence, "detach"):
                    sequence = sequence.detach().cpu().numpy()
                batches.append(np.asarray(sequence).tolist())
            return batches
        return [np.asarray(token_ids).tolist()]

    array = np.asarray(token_ids)
    if array.ndim == 0:
        return []
    if array.ndim == 1:
        return [array.tolist()]
    return array.tolist()


def _decode_token_id_strings(
    tokenizer: Any,
    token_ids: Any,
    *,
    pad_token_id: int | None = None,
    max_token_id: int | None = None,
) -> list[str]:
    """Decode token ids into normalized source strings."""
    resolved_pad_token_id = (
        _tokenizer_pad_token_id(tokenizer) if pad_token_id is None else pad_token_id
    )
    resolved_max_token_id = (
        _tokenizer_max_token_id(tokenizer) if max_token_id is None else max_token_id
    )
    token_id_batches = _coerce_token_id_batches(token_ids)
    if not token_id_batches:
        return []

    sanitized_batches = [
        _sanitize_token_ids_for_decoding(
            np.asarray(sequence),
            pad_token_id=resolved_pad_token_id,
            max_token_id=resolved_max_token_id,
        ).tolist()
        for sequence in token_id_batches
    ]
    decoded = tokenizer.batch_decode(sanitized_batches, skip_special_tokens=True)
    return [_normalize_decoded_text(text) for text in decoded]


def _dataset_column_values(dataset: Any, column: str) -> Any | None:
    """Return one column from common dataset containers when available."""
    raw_features = getattr(dataset, "features", None)
    if isinstance(raw_features, dict) and column in raw_features:
        values = raw_features[column]
        if isinstance(values, list | tuple):
            return values
    if isinstance(dataset, MappingABC) and column in dataset:
        return dataset[column]
    if hasattr(dataset, "columns") and column in dataset.columns:
        return dataset[column].tolist()
    if hasattr(dataset, "column_names") and column in dataset.column_names:
        return dataset[column]
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        values = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            if not isinstance(item, MappingABC) or column not in item:
                return None
            values.append(item[column])
        return values
    return None


def _is_real_training_source_id(source_id: str) -> bool:
    """Return whether a source id should count as a real source dataset."""
    normalized = source_id.strip()
    if not normalized:
        return False
    return not normalized.startswith(
        (
            "synthetic_multicod",
            "synthetic_pretrain_multicod",
            "masterlist_",
        )
    )


def collect_label_source_index(
    dataset: Any,
    label_column: str,
    label_separator: str,
    source_column: str = "source_id",
) -> dict[str, set[str]]:
    """Collect source datasets observed for each training label."""
    label_values = _dataset_column_values(dataset, label_column)
    source_values = _dataset_column_values(dataset, source_column)
    if label_values is None or source_values is None:
        return {}

    label_list = list(label_values)
    source_list = list(source_values)
    if len(label_list) != len(source_list):
        return {}

    index: dict[str, set[str]] = {}
    for value, source_value in zip(label_list, source_list):
        source_id = str(source_value).strip()
        if not _is_real_training_source_id(source_id):
            continue
        for label in _split_label_value(value, label_separator):
            index.setdefault(label, set()).add(source_id)
    return index


def collect_input_strings(
    dataset: Any,
    tokenizer: Any,
    input_ids_column: str = "input_ids",
) -> set[str]:
    """Collect normalized decoded input strings from a tokenized dataset."""
    input_ids = _dataset_column_values(dataset, input_ids_column)
    if input_ids is None:
        return set()
    return {
        text for text in _decode_token_id_strings(tokenizer, input_ids) if text.strip()
    }


def _micro_precision_recall_f1(
    predictions: list[set[str]], labels: list[set[str]]
) -> dict[str, float]:
    """Compute micro-averaged precision/recall/F1/Jaccard over per-example code sets."""
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
    jaccard = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "micro_jaccard": jaccard,
    }


def _sample_precision_recall_f1(
    predictions: list[set[str]], labels: list[set[str]]
) -> dict[str, float]:
    """Compute sample-averaged set overlap metrics for multi-label predictions."""
    if not predictions:
        return {
            "sample_precision": 0.0,
            "sample_recall": 0.0,
            "sample_f1": 0.0,
            "sample_jaccard": 0.0,
        }

    per_sample_precision = []
    per_sample_recall = []
    per_sample_f1 = []
    per_sample_jaccard = []
    for predicted_codes, label_codes in zip(predictions, labels):
        tp = len(predicted_codes.intersection(label_codes))
        pred_total = len(predicted_codes)
        label_total = len(label_codes)
        union_total = len(predicted_codes.union(label_codes))

        precision = tp / pred_total if pred_total > 0 else float(label_total == 0)
        recall = tp / label_total if label_total > 0 else float(pred_total == 0)
        f1 = (
            (2 * tp) / (pred_total + label_total)
            if (pred_total + label_total) > 0
            else 1.0
        )
        jaccard = tp / union_total if union_total > 0 else 1.0

        per_sample_precision.append(precision)
        per_sample_recall.append(recall)
        per_sample_f1.append(f1)
        per_sample_jaccard.append(jaccard)

    return {
        "sample_precision": float(np.mean(per_sample_precision)),
        "sample_recall": float(np.mean(per_sample_recall)),
        "sample_f1": float(np.mean(per_sample_f1)),
        "sample_jaccard": float(np.mean(per_sample_jaccard)),
    }


def _multilabel_diagnostic_metrics(
    predictions: list[set[str]],
    labels: list[set[str]],
    label_universe: set[str] | None = None,
) -> dict[str, float]:
    """Compute multi-label diagnostics for label count and per-label errors."""
    observed_classes = (
        set().union(*predictions, *labels) if predictions or labels else set()
    )
    effective_universe = set(label_universe or set()) | observed_classes
    sample_count = len(predictions)

    false_positives = []
    false_negatives = []
    predicted_counts = []
    label_counts = []
    label_count_errors = []
    hamming_errors = 0
    for predicted_codes, label_codes in zip(predictions, labels):
        false_positive_count = len(predicted_codes.difference(label_codes))
        false_negative_count = len(label_codes.difference(predicted_codes))
        predicted_count = len(predicted_codes)
        label_count = len(label_codes)

        false_positives.append(false_positive_count)
        false_negatives.append(false_negative_count)
        predicted_counts.append(predicted_count)
        label_counts.append(label_count)
        label_count_errors.append(abs(predicted_count - label_count))
        hamming_errors += false_positive_count + false_negative_count

    hamming_denominator = sample_count * len(effective_universe)
    hamming_loss = (
        float(hamming_errors / hamming_denominator) if hamming_denominator > 0 else 0.0
    )
    empty_prediction_count = sum(
        1 for predicted_codes in predictions if not predicted_codes
    )

    return {
        "hamming_loss": hamming_loss,
        "hamming_score": 1.0 - hamming_loss,
        "avg_predicted_label_count": float(np.mean(predicted_counts))
        if predicted_counts
        else 0.0,
        "avg_true_label_count": float(np.mean(label_counts)) if label_counts else 0.0,
        "label_count_mae": float(np.mean(label_count_errors))
        if label_count_errors
        else 0.0,
        "avg_false_positives_per_sample": float(np.mean(false_positives))
        if false_positives
        else 0.0,
        "avg_false_negatives_per_sample": float(np.mean(false_negatives))
        if false_negatives
        else 0.0,
        "empty_prediction_rate": float(empty_prediction_count / sample_count)
        if sample_count > 0
        else 0.0,
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
    all_classes = (
        classes
        if classes is not None
        else (set(class_label_total) | set(class_pred_total))
    )
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
    class_tp, class_label_total, class_pred_total = _per_class_stats(
        predictions, labels
    )
    raw = _macro_from_class_stats(class_tp, class_label_total, class_pred_total)
    return {f"macro_{k}": v for k, v in raw.items()}


def _chapter_block_codes(codes: set[str]) -> set[str]:
    """Collapse ICD10h codes to their first three-character chapter block."""
    return {code.strip()[:3] for code in codes if code.strip()}


def _chapter_block_metric_set(
    predictions: list[set[str]],
    labels: list[set[str]],
    *,
    multi_label: bool,
    label_universe: set[str] | None = None,
) -> dict[str, float]:
    """Compute standard metrics after collapsing codes to chapter blocks."""
    block_predictions = [_chapter_block_codes(codes) for codes in predictions]
    block_labels = [_chapter_block_codes(codes) for codes in labels]
    block_matches = [
        prediction == label
        for prediction, label in zip(block_predictions, block_labels)
    ]
    block_universe = (
        _chapter_block_codes(label_universe) if label_universe is not None else None
    )
    metrics = _prediction_metric_set(
        block_predictions,
        block_labels,
        block_matches,
        multi_label=multi_label,
        label_universe=block_universe,
        include_chapter_block=False,
    )
    return {f"chapter_block_{key}": value for key, value in metrics.items()}


def _prediction_metric_set(
    predictions: list[set[str]],
    labels: list[set[str]],
    matches: list[bool],
    *,
    multi_label: bool,
    label_universe: set[str] | None = None,
    include_chapter_block: bool = True,
) -> dict[str, float]:
    """Compute the standard metric set for one prediction slice."""
    accuracy = float(np.mean(matches)) if matches else 0.0
    result: dict[str, float] = {"accuracy": accuracy, "exact_match": accuracy}
    if multi_label:
        result.update(_micro_precision_recall_f1(predictions, labels))
        result.update(_sample_precision_recall_f1(predictions, labels))
        result.update(
            _multilabel_diagnostic_metrics(
                predictions,
                labels,
                label_universe=label_universe,
            )
        )
    result.update(_macro_precision_recall_f1(predictions, labels))
    if include_chapter_block:
        result.update(
            _chapter_block_metric_set(
                predictions,
                labels,
                multi_label=multi_label,
                label_universe=label_universe,
            )
        )
    return result


def _prefixed_metrics(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    """Prefix metric keys with a bucket name."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _subset_by_indices(values: list[Any], indices: list[int]) -> list[Any]:
    """Return a list subset for the requested integer indices."""
    return [values[index] for index in indices]


def _hierarchy_truncated_codes(
    code_sets: list[set[str]],
    length: int,
) -> list[set[str]]:
    """Return code sets truncated to the first `length` characters of each code."""
    return [
        {code[:length] for code in codes if code}
        for codes in code_sets
    ]


def _hierarchy_metrics(
    predictions: list[set[str]],
    labels: list[set[str]],
    *,
    multi_label: bool,
) -> dict[str, float]:
    """Compute accuracy/F1 metrics at chapter (1 char) and block (3 chars) levels.

    ICD10h codes are tree-structured (`^[A-Z]\\d{2}\\.\\d{3}$`); predicting J18.0
    when the gold is J18.9 is much closer than predicting K76.4. Reporting at
    coarser levels gives a more honest picture of model quality for downstream
    historian users.
    """
    result: dict[str, float] = {}
    for prefix, length in (("chapter", 1), ("block", 3)):
        truncated_predictions = _hierarchy_truncated_codes(predictions, length)
        truncated_labels = _hierarchy_truncated_codes(labels, length)
        truncated_matches = [
            prediction == label
            for prediction, label in zip(truncated_predictions, truncated_labels)
        ]
        bucket_metrics = _prediction_metric_set(
            truncated_predictions,
            truncated_labels,
            truncated_matches,
            multi_label=multi_label,
        )
        result.update(_prefixed_metrics(bucket_metrics, prefix))
    return result


def _per_source_metrics(
    predictions: list[set[str]],
    labels: list[set[str]],
    matches: list[bool],
    source_ids: list[str] | None,
    *,
    multi_label: bool,
    label_universe: set[str] | None = None,
    excluded_sources: frozenset[str] = _PER_SOURCE_AGGREGATE_EXCLUDED,
) -> dict[str, float]:
    """Compute the standard metric set for each source_id slice of predictions.

    Sources listed in ``excluded_sources`` are skipped entirely so the W&B
    Compare-runs view does not surface them as comparable data points.
    historic_strings_en_2024 is the canonical example: it is the masterlist's
    English-translated reference, not historical messy archive text, so its
    accuracy is mechanically inflated and should not be plotted alongside the
    five real archival sources.
    """
    if source_ids is None or len(source_ids) != len(predictions):
        return {}

    indices_by_source: dict[str, list[int]] = {}
    for index, source in enumerate(source_ids):
        indices_by_source.setdefault(str(source), []).append(index)

    result: dict[str, float] = {}
    for source, indices in indices_by_source.items():
        if source in excluded_sources:
            continue
        bucket_metrics = _prediction_metric_set(
            _subset_by_indices(predictions, indices),
            _subset_by_indices(labels, indices),
            _subset_by_indices(matches, indices),
            multi_label=multi_label,
            label_universe=label_universe,
        )
        sanitized = _sanitize_source_key(source)
        prefix = f"{_PER_SOURCE_METRIC_PREFIX}{sanitized}"
        result[f"{prefix}_count"] = float(len(indices))
        result.update(_prefixed_metrics(bucket_metrics, prefix))
    return result


def _seen_unseen_string_metrics(
    predictions: list[set[str]],
    labels: list[set[str]],
    matches: list[bool],
    input_strings: list[str] | None,
    train_input_strings: set[str],
    *,
    multi_label: bool,
    label_universe: set[str] | None = None,
) -> dict[str, float]:
    """Compute standard metrics split by whether each input string appeared in train."""
    if input_strings is None or len(input_strings) != len(predictions):
        return {}

    seen_indices = [
        index
        for index, input_text in enumerate(input_strings)
        if input_text in train_input_strings
    ]
    unseen_indices = [
        index
        for index, input_text in enumerate(input_strings)
        if input_text not in train_input_strings
    ]
    total_count = len(input_strings)
    result: dict[str, float] = {
        "seen_string_count": float(len(seen_indices)),
        "unseen_string_count": float(len(unseen_indices)),
        "seen_string_rate": float(len(seen_indices) / total_count)
        if total_count > 0
        else 0.0,
        "unseen_string_rate": float(len(unseen_indices) / total_count)
        if total_count > 0
        else 0.0,
    }

    for prefix, indices in (("seen", seen_indices), ("unseen", unseen_indices)):
        bucket_metrics = _prediction_metric_set(
            _subset_by_indices(predictions, indices),
            _subset_by_indices(labels, indices),
            _subset_by_indices(matches, indices),
            multi_label=multi_label,
            label_universe=label_universe,
        )
        result.update(_prefixed_metrics(bucket_metrics, prefix))

    return result


def _source_transfer_label_metrics(
    predictions: list[set[str]],
    labels: list[set[str]],
    matches: list[bool],
    source_ids: Sequence[str],
    train_label_sources: Mapping[str, set[str]],
    *,
    multi_label: bool,
    label_universe: set[str] | None = None,
) -> dict[str, float]:
    """Compute metrics for labels that require source-to-source transfer."""
    if len(source_ids) != len(predictions):
        return {}
    if not train_label_sources:
        return {}

    bucket_indices: dict[str, list[int]] = {
        "source_transfer_label": [],
        "same_source_label": [],
        "unseen_label": [],
    }
    bucket_true_counts = {key: 0 for key in bucket_indices}
    bucket_hits = {key: 0 for key in bucket_indices}

    for index, (predicted_codes, label_codes, source_id) in enumerate(
        zip(predictions, labels, source_ids)
    ):
        sample_buckets = set()
        normalized_source_id = str(source_id).strip()
        for label in label_codes:
            training_sources = train_label_sources.get(label, set())
            if not training_sources:
                bucket = "unseen_label"
            elif normalized_source_id in training_sources:
                bucket = "same_source_label"
            else:
                bucket = "source_transfer_label"

            bucket_true_counts[bucket] += 1
            if label in predicted_codes:
                bucket_hits[bucket] += 1
            sample_buckets.add(bucket)

        for bucket in sample_buckets:
            bucket_indices[bucket].append(index)

    sample_count = len(predictions)
    result: dict[str, float] = {}
    for bucket, indices in bucket_indices.items():
        true_count = bucket_true_counts[bucket]
        result[f"{bucket}_sample_count"] = float(len(indices))
        result[f"{bucket}_sample_rate"] = (
            float(len(indices) / sample_count) if sample_count > 0 else 0.0
        )
        result[f"{bucket}_true_count"] = float(true_count)
        result[f"{bucket}_recall"] = (
            float(bucket_hits[bucket] / true_count) if true_count > 0 else 0.0
        )
        bucket_metrics = _prediction_metric_set(
            _subset_by_indices(predictions, indices),
            _subset_by_indices(labels, indices),
            _subset_by_indices(matches, indices),
            multi_label=multi_label,
            label_universe=label_universe,
        )
        result.update(_prefixed_metrics(bucket_metrics, bucket))
    return result


def _extract_eval_prediction(eval_pred: Any) -> tuple[Any, Any, Any | None]:
    """Extract predictions, labels, and optional metric inputs from Trainer output."""
    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        return (
            eval_pred.predictions,
            eval_pred.label_ids,
            getattr(eval_pred, "inputs", None),
        )

    values = tuple(eval_pred)
    if len(values) < 2:
        raise ValueError("Metric callbacks require predictions and label ids.")
    metric_inputs = values[2] if len(values) > 2 else None
    return values[0], values[1], metric_inputs


def build_exact_match_accuracy_metric(
    tokenizer: Any,
    label_separator: str = ",",
    max_label_count: int = 1,
    train_classes: set[str] | None = None,
    train_label_sources: Mapping[str, set[str]] | None = None,
    train_input_strings: set[str] | None = None,
    artifact_logger: MetricArtifactLogger | None = None,
    metric_mode: str = "all",
    save_metric: str | None = None,
) -> Callable[[Any], dict[str, float]]:
    """Build a compute_metrics callback with exact-match and overlap metrics.

    When max_label_count is 1 (single-label), micro metrics are skipped because
    they are mathematically identical to accuracy in that regime.

    When train_input_strings is provided and metric inputs are available, seen/unseen
    metrics are split by whether the decoded source string appeared in training.
    """
    pad_token_id = _tokenizer_pad_token_id(tokenizer)
    max_token_id = _tokenizer_max_token_id(tokenizer)
    multi_label = max_label_count > 1

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Compute exact-match accuracy and class-level metrics."""
        predictions, labels, metric_inputs = _extract_eval_prediction(eval_pred)

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
        predicted_code_sets = [
            _split_predicted_codes(text, label_separator)
            for text in normalized_predictions
        ]
        label_code_sets = [
            _split_predicted_codes(text, label_separator) for text in normalized_labels
        ]
        if multi_label:
            matches = [
                prediction == label
                for prediction, label in zip(predicted_code_sets, label_code_sets)
            ]
        else:
            matches = [
                prediction == label
                for prediction, label in zip(normalized_predictions, normalized_labels)
            ]

        result = _prediction_metric_set(
            predicted_code_sets,
            label_code_sets,
            matches,
            multi_label=multi_label,
            label_universe=train_classes,
        )
        result.update(
            _hierarchy_metrics(
                predicted_code_sets,
                label_code_sets,
                multi_label=multi_label,
            )
        )
        result.update(
            _per_source_metrics(
                predicted_code_sets,
                label_code_sets,
                matches,
                current_metric_source_ids(),
                multi_label=multi_label,
                label_universe=train_classes,
            )
        )
        input_strings = (
            _decode_token_id_strings(
                tokenizer,
                metric_inputs,
                pad_token_id=pad_token_id,
                max_token_id=max_token_id,
            )
            if metric_inputs is not None
            else None
        )
        if train_input_strings is not None:
            result.update(
                _seen_unseen_string_metrics(
                    predicted_code_sets,
                    label_code_sets,
                    matches,
                    input_strings,
                    train_input_strings,
                    multi_label=multi_label,
                    label_universe=train_classes,
                )
            )
        source_ids = current_metric_source_ids()
        if train_label_sources is not None and source_ids is not None:
            result.update(
                _source_transfer_label_metrics(
                    predicted_code_sets,
                    label_code_sets,
                    matches,
                    source_ids,
                    train_label_sources,
                    multi_label=multi_label,
                    label_universe=train_classes,
                )
            )
        if artifact_logger is not None:
            artifact_logger(
                metric_scope=current_metric_artifact_scope(),
                input_strings=input_strings,
                predictions=normalized_predictions,
                labels=normalized_labels,
                metrics=result,
            )
        return filter_metrics_for_logging(
            result,
            mode=metric_mode,
            save_metric=save_metric,
        )

    return compute_metrics


def build_sequence_classification_metric(
    id2label: Mapping[int, str],
    tokenizer: Any | None = None,
    train_label_sources: Mapping[str, set[str]] | None = None,
    train_input_strings: set[str] | None = None,
    artifact_logger: MetricArtifactLogger | None = None,
    metric_mode: str = "all",
    save_metric: str | None = None,
) -> Callable[[Any], dict[str, float]]:
    """Build compute_metrics callback for single-label sequence classification."""
    normalized_id2label = {int(key): str(value) for key, value in id2label.items()}
    pad_token_id = _tokenizer_pad_token_id(tokenizer) if tokenizer is not None else 0
    max_token_id = _tokenizer_max_token_id(tokenizer) if tokenizer is not None else None

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        """Compute accuracy and macro precision/recall/F1 from classifier logits."""
        predictions, labels, metric_inputs = _extract_eval_prediction(eval_pred)

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

        predicted_code_sets = [{label} for label in normalized_predictions]
        label_code_sets = [{label} for label in normalized_labels]
        result = _prediction_metric_set(
            predicted_code_sets,
            label_code_sets,
            matches,
            multi_label=False,
        )
        result.update(
            _hierarchy_metrics(
                predicted_code_sets,
                label_code_sets,
                multi_label=False,
            )
        )
        result.update(
            _per_source_metrics(
                predicted_code_sets,
                label_code_sets,
                matches,
                current_metric_source_ids(),
                multi_label=False,
            )
        )
        input_strings = (
            (
                _decode_token_id_strings(
                    tokenizer,
                    metric_inputs,
                    pad_token_id=pad_token_id,
                    max_token_id=max_token_id,
                )
                if metric_inputs is not None
                else None
            )
            if tokenizer is not None
            else None
        )
        if train_input_strings is not None and tokenizer is not None:
            result.update(
                _seen_unseen_string_metrics(
                    predicted_code_sets,
                    label_code_sets,
                    matches,
                    input_strings,
                    train_input_strings,
                    multi_label=False,
                )
            )
        source_ids = current_metric_source_ids()
        if train_label_sources is not None and source_ids is not None:
            result.update(
                _source_transfer_label_metrics(
                    predicted_code_sets,
                    label_code_sets,
                    matches,
                    source_ids,
                    train_label_sources,
                    multi_label=False,
                )
            )
        if artifact_logger is not None:
            artifact_logger(
                metric_scope=current_metric_artifact_scope(),
                input_strings=input_strings,
                predictions=normalized_predictions,
                labels=normalized_labels,
                metrics=result,
            )
        return filter_metrics_for_logging(
            result,
            mode=metric_mode,
            save_metric=save_metric,
        )

    return compute_metrics
