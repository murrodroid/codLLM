import numpy as np
import pytest

from codllm.metrics import (
    _macro_precision_recall_f1,
    _micro_precision_recall_f1,
    _multilabel_diagnostic_metrics,
    _sample_precision_recall_f1,
    build_exact_match_accuracy_metric,
    collect_label_classes,
)


class TestMicroPrecisionRecallF1:
    def test_perfect_predictions(self) -> None:
        preds = [{"A", "B"}, {"C"}]
        labels = [{"A", "B"}, {"C"}]
        result = _micro_precision_recall_f1(preds, labels)
        assert result["micro_precision"] == 1.0
        assert result["micro_recall"] == 1.0
        assert result["micro_f1"] == 1.0

    def test_no_overlap(self) -> None:
        preds = [{"X"}]
        labels = [{"Y"}]
        result = _micro_precision_recall_f1(preds, labels)
        assert result["micro_precision"] == 0.0
        assert result["micro_recall"] == 0.0
        assert result["micro_f1"] == 0.0

    def test_partial_overlap(self) -> None:
        preds = [{"A", "B", "C"}]
        labels = [{"A", "B", "D"}]
        # tp=2, fp=1, fn=1
        result = _micro_precision_recall_f1(preds, labels)
        assert result["micro_precision"] == pytest.approx(2 / 3)
        assert result["micro_recall"] == pytest.approx(2 / 3)
        assert result["micro_f1"] == pytest.approx(2 / 3)
        assert result["micro_jaccard"] == pytest.approx(0.5)

    def test_empty_inputs(self) -> None:
        result = _micro_precision_recall_f1([], [])
        assert result["micro_precision"] == 0.0
        assert result["micro_recall"] == 0.0
        assert result["micro_f1"] == 0.0

    def test_empty_prediction_set(self) -> None:
        preds = [set()]
        labels = [{"A"}]
        # tp=0, fp=0, fn=1
        result = _micro_precision_recall_f1(preds, labels)
        assert result["micro_precision"] == 0.0
        assert result["micro_recall"] == 0.0
        assert result["micro_f1"] == 0.0

    def test_extra_predictions(self) -> None:
        preds = [{"A", "B", "C"}]
        labels = [{"A"}]
        # tp=1, fp=2, fn=0
        result = _micro_precision_recall_f1(preds, labels)
        assert result["micro_precision"] == pytest.approx(1 / 3)
        assert result["micro_recall"] == 1.0


def test_collect_label_classes_splits_string_and_list_values() -> None:
    """Label class collection should support separated strings and list values."""
    dataset = {"label": ["A00 | A01", ["B00", "C00"], ""]}

    result = collect_label_classes(
        dataset,
        label_column="label",
        label_separator=" | ",
    )

    assert result == {"A00", "A01", "B00", "C00"}


class TestSamplePrecisionRecallF1:
    def test_partial_overlap(self) -> None:
        """Sample metrics should average overlap quality per row."""
        preds = [{"A", "B"}, {"C"}, set()]
        labels = [{"A", "C"}, {"C", "D"}, set()]

        result = _sample_precision_recall_f1(preds, labels)

        assert result["sample_precision"] == pytest.approx(5 / 6)
        assert result["sample_recall"] == pytest.approx(2 / 3)
        assert result["sample_f1"] == pytest.approx(13 / 18)
        assert result["sample_jaccard"] == pytest.approx(11 / 18)

    def test_empty_inputs(self) -> None:
        """Sample metrics should return zero for an empty evaluation batch."""
        result = _sample_precision_recall_f1([], [])

        assert result["sample_precision"] == 0.0
        assert result["sample_recall"] == 0.0
        assert result["sample_f1"] == 0.0
        assert result["sample_jaccard"] == 0.0


class TestMultilabelDiagnosticMetrics:
    def test_hamming_and_label_count_metrics(self) -> None:
        """Diagnostics should summarize per-label errors and cardinality drift."""
        preds = [{"A", "B"}, {"C"}]
        labels = [{"A", "C"}, {"C", "D"}]

        result = _multilabel_diagnostic_metrics(
            preds,
            labels,
            label_universe={"A", "B", "C", "D", "E"},
        )

        assert result["hamming_loss"] == pytest.approx(0.3)
        assert result["hamming_score"] == pytest.approx(0.7)
        assert result["avg_predicted_label_count"] == pytest.approx(1.5)
        assert result["avg_true_label_count"] == pytest.approx(2.0)
        assert result["label_count_mae"] == pytest.approx(0.5)
        assert result["avg_false_positives_per_sample"] == pytest.approx(0.5)
        assert result["avg_false_negatives_per_sample"] == pytest.approx(1.0)
        assert result["empty_prediction_rate"] == 0.0


class TestMacroPrecisionRecallF1:
    def test_perfect_predictions(self) -> None:
        preds = [{"A", "B"}, {"C"}]
        labels = [{"A", "B"}, {"C"}]
        result = _macro_precision_recall_f1(preds, labels)
        assert result["macro_precision"] == 1.0
        assert result["macro_recall"] == 1.0
        assert result["macro_f1"] == 1.0

    def test_no_overlap(self) -> None:
        preds = [{"X"}]
        labels = [{"Y"}]
        # class X: tp=0, pred=1, label=0 -> p=0, r=0
        # class Y: tp=0, pred=0, label=1 -> p=0, r=0
        result = _macro_precision_recall_f1(preds, labels)
        assert result["macro_precision"] == 0.0
        assert result["macro_recall"] == 0.0
        assert result["macro_f1"] == 0.0

    def test_empty_inputs(self) -> None:
        result = _macro_precision_recall_f1([], [])
        assert result["macro_precision"] == 0.0
        assert result["macro_recall"] == 0.0
        assert result["macro_f1"] == 0.0

    def test_imbalanced_classes(self) -> None:
        """Macro averaging should weight rare and frequent classes equally."""
        # class A: appears 3 times in labels, predicted correctly 3 times -> r=1.0
        # class B: appears 1 time in labels, predicted correctly 0 times -> r=0.0
        preds = [{"A"}, {"A"}, {"A"}, {"A"}]
        labels = [{"A"}, {"A"}, {"A"}, {"B"}]
        result = _macro_precision_recall_f1(preds, labels)
        # macro recall = mean(1.0, 0.0) = 0.5
        assert result["macro_recall"] == pytest.approx(0.5)
        # class A: tp=3, pred=4, label=3 -> p=3/4
        # class B: tp=0, pred=0, label=1 -> p=0
        assert result["macro_precision"] == pytest.approx(3 / 8)

    def test_predicted_class_not_in_labels(self) -> None:
        """A predicted class absent from labels should count toward precision."""
        preds = [{"A", "Z"}]
        labels = [{"A"}]
        # class A: tp=1, pred=1, label=1 -> p=1, r=1, f=1
        # class Z: tp=0, pred=1, label=0 -> p=0, r=0, f=0
        result = _macro_precision_recall_f1(preds, labels)
        assert result["macro_precision"] == pytest.approx(0.5)
        assert result["macro_recall"] == pytest.approx(0.5)
        assert result["macro_f1"] == pytest.approx(0.5)

    def test_single_class_perfect(self) -> None:
        preds = [{"A"}, {"A"}]
        labels = [{"A"}, {"A"}]
        result = _macro_precision_recall_f1(preds, labels)
        assert result["macro_precision"] == 1.0
        assert result["macro_recall"] == 1.0
        assert result["macro_f1"] == 1.0


class _StrictDecodeTokenizer:
    """Tokenizer stub that raises on invalid token ids."""

    pad_token_id = 0
    vocab_size = 100

    def batch_decode(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode by validating token ids and joining them into strings."""
        decoded: list[str] = []
        for sequence in sequences:
            for token_id in sequence:
                if token_id < 0 or token_id > (2**32 - 1):
                    raise OverflowError(
                        "out of range integral type conversion attempted"
                    )
            if skip_special_tokens:
                kept = [
                    token_id for token_id in sequence if token_id != self.pad_token_id
                ]
            else:
                kept = sequence
            decoded.append(" ".join(str(token_id) for token_id in kept))
        return decoded


class _EvalPrediction:
    """Minimal EvalPrediction stub with optional metric inputs."""

    def __init__(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray,
        inputs: np.ndarray,
    ) -> None:
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs


def test_build_exact_match_accuracy_metric_sanitizes_invalid_prediction_ids() -> None:
    """Metric callback should sanitize invalid prediction ids before decoding."""
    metric_fn = build_exact_match_accuracy_metric(_StrictDecodeTokenizer())
    predictions = np.array(
        [
            [1, -100, 2],
            [3, float("nan"), 0],
            [4, 2**40, 0],
        ],
        dtype=np.float64,
    )
    labels = np.array(
        [
            [1, -100, 2],
            [3, -100, 0],
            [4, -100, 0],
        ],
        dtype=np.int64,
    )

    metrics = metric_fn((predictions, labels))

    assert "accuracy" in metrics


def test_multilabel_accuracy_ignores_code_order() -> None:
    """Multi-label exact-set accuracy should not depend on decoded code order."""
    metric_fn = build_exact_match_accuracy_metric(
        _StrictDecodeTokenizer(),
        label_separator=" ",
        max_label_count=2,
    )
    predictions = np.array([[1, 2, 0]], dtype=np.int64)
    labels = np.array([[2, 1, -100]], dtype=np.int64)

    metrics = metric_fn((predictions, labels))

    assert metrics["accuracy"] == 1.0


def test_seen_unseen_metrics_use_input_strings_not_label_classes() -> None:
    """Seen/unseen buckets should be based on source strings from train."""
    metric_fn = build_exact_match_accuracy_metric(
        _StrictDecodeTokenizer(),
        train_classes={"1", "2"},
        train_input_strings={"10"},
    )
    predictions = np.array([[1, 0], [1, 0]], dtype=np.int64)
    labels = np.array([[1, -100], [2, -100]], dtype=np.int64)
    inputs = np.array([[10, 0], [30, 0]], dtype=np.int64)

    metrics = metric_fn(_EvalPrediction(predictions, labels, inputs))

    assert metrics["seen_string_count"] == 1.0
    assert metrics["unseen_string_count"] == 1.0
    assert metrics["seen_accuracy"] == 1.0
    assert metrics["unseen_accuracy"] == 0.0
    assert "seen_class_count" not in metrics


def test_multilabel_metric_callback_reports_sample_and_hamming_metrics() -> None:
    """Multi-label metric callback should expose exact, sample, and Hamming metrics."""
    metric_fn = build_exact_match_accuracy_metric(
        _StrictDecodeTokenizer(),
        label_separator=" ",
        max_label_count=3,
    )
    predictions = np.array(
        [
            [1, 2, 0],
            [3, 0, 0],
            [4, 5, 0],
        ],
        dtype=np.int64,
    )
    labels = np.array(
        [
            [1, 3, -100],
            [3, -100, -100],
            [5, 4, -100],
        ],
        dtype=np.int64,
    )

    metrics = metric_fn((predictions, labels))

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["exact_match"] == pytest.approx(2 / 3)
    assert metrics["micro_precision"] == pytest.approx(4 / 5)
    assert metrics["micro_recall"] == pytest.approx(4 / 5)
    assert metrics["micro_f1"] == pytest.approx(4 / 5)
    assert metrics["micro_jaccard"] == pytest.approx(2 / 3)
    assert metrics["sample_f1"] == pytest.approx(5 / 6)
    assert metrics["sample_jaccard"] == pytest.approx(7 / 9)
    assert metrics["hamming_loss"] == pytest.approx(2 / 15)
    assert metrics["hamming_score"] == pytest.approx(13 / 15)
    assert metrics["label_count_mae"] == 0.0
