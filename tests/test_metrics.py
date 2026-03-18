import pytest

from codllm.metrics import _micro_precision_recall_f1, _macro_precision_recall_f1


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
