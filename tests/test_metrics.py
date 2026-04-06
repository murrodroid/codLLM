import numpy as np
import pytest

from codllm.metrics import (
    _macro_precision_recall_f1,
    _micro_precision_recall_f1,
    build_exact_match_accuracy_metric,
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
                    raise OverflowError("out of range integral type conversion attempted")
            if skip_special_tokens:
                kept = [token_id for token_id in sequence if token_id != self.pad_token_id]
            else:
                kept = sequence
            decoded.append(" ".join(str(token_id) for token_id in kept))
        return decoded


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
