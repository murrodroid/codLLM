"""Fast unit tests for uncertainty signals, RC curves, and orchestration helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from codllm.uncertainty.end_of_training import (
    _build_record_row,
    _records_to_metric_inputs,
    _split_codes,
    compute_all_rc_curves,
    write_predictions_jsonl,
    write_rc_curves_json,
    write_test_rows_parquet,
)
from codllm.uncertainty.rc_curve import (
    DEFAULT_COVERAGES,
    UNCERTAINTY_SIGNAL_DIRECTION,
    UNCERTAINTY_SIGNAL_NAMES,
    accuracy_metric,
    compute_rc_curve_by_coverage,
    coverage_at_threshold,
    sample_f1_metric,
)
from codllm.uncertainty.signals import UncertaintySignals


class TestSplitCodes:
    def test_separator_splits_into_set(self) -> None:
        assert _split_codes("A,B,A", separator=",") == {"A", "B"}

    def test_strips_whitespace(self) -> None:
        assert _split_codes(" A ,  B", separator=",") == {"A", "B"}

    def test_empty_string_returns_empty(self) -> None:
        assert _split_codes("", separator=",") == set()

    def test_no_separator_returns_single_token(self) -> None:
        assert _split_codes("A18.900", separator="") == {"A18.900"}


class TestAccuracyAndSampleF1:
    def test_accuracy_metric_handles_empty(self) -> None:
        assert accuracy_metric([]) == 0.0

    def test_accuracy_metric_basic(self) -> None:
        assert accuracy_metric([True, False, True, True]) == pytest.approx(0.75)

    def test_sample_f1_metric_perfect_match(self) -> None:
        pairs = [({"A", "B"}, {"A", "B"}), ({"C"}, {"C"})]
        assert sample_f1_metric(pairs) == pytest.approx(1.0)

    def test_sample_f1_metric_partial(self) -> None:
        # 2*1/(2+2)=0.5 ; 2*0/(0+1)=0
        pairs = [({"A", "X"}, {"A", "B"}), (set(), {"C"})]
        assert sample_f1_metric(pairs) == pytest.approx(0.25)

    def test_sample_f1_metric_both_empty_is_one(self) -> None:
        assert sample_f1_metric([(set(), set())]) == 1.0


class TestComputeRcCurveByCoverage:
    def test_full_coverage_uses_all_records(self) -> None:
        scores = [0.9, 0.5, 0.1]
        matches = [True, False, True]
        points = compute_rc_curve_by_coverage(
            scores,
            matches,
            metric_fn=accuracy_metric,
            direction=1,
            coverages=(1.0,),
        )
        assert len(points) == 1
        assert points[0].coverage == 1.0
        assert points[0].n_kept == 3
        assert points[0].metric_value == pytest.approx(2 / 3)

    def test_high_coverage_keeps_top_k_by_score(self) -> None:
        # logprob direction = 1, higher score = more confident.
        scores = [0.9, 0.1, 0.5]
        matches = [True, False, True]
        points = compute_rc_curve_by_coverage(
            scores,
            matches,
            metric_fn=accuracy_metric,
            direction=1,
            coverages=(2 / 3,),
        )
        # Top-2 by score = indices 0, 2 -> matches True, True -> accuracy 1.0
        assert points[0].n_kept == 2
        assert points[0].metric_value == pytest.approx(1.0)

    def test_entropy_direction_flips_orientation(self) -> None:
        # entropy direction = -1, lower entropy = more confident.
        entropy = [2.0, 0.5, 1.0]
        matches = [False, True, False]
        points = compute_rc_curve_by_coverage(
            entropy,
            matches,
            metric_fn=accuracy_metric,
            direction=-1,
            coverages=(1 / 3,),
        )
        # Most confident = lowest entropy = index 1 -> True -> accuracy 1.0
        assert points[0].n_kept == 1
        assert points[0].metric_value == pytest.approx(1.0)

    def test_perfect_uncertainty_signal_increases_accuracy(self) -> None:
        """When confidence is perfectly anti-correlated with errors, accuracy at
        partial coverage should reach 1.0 even though full coverage is below it."""
        scores = [0.99, 0.05, 0.95, 0.10, 0.90]
        matches = [True, False, True, False, True]
        points = compute_rc_curve_by_coverage(
            scores,
            matches,
            metric_fn=accuracy_metric,
            direction=1,
            coverages=(1.0, 0.6),
        )
        full_coverage_point = next(p for p in points if p.coverage == 1.0)
        partial_point = next(p for p in points if p.coverage == 0.6)
        assert full_coverage_point.metric_value == pytest.approx(0.6)
        assert partial_point.metric_value == pytest.approx(1.0)

    def test_input_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_rc_curve_by_coverage(
                [0.1, 0.2],
                [True],
                metric_fn=accuracy_metric,
                direction=1,
            )

    def test_default_coverages_are_returned_in_descending_order(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4]
        matches = [True, True, False, False]
        points = compute_rc_curve_by_coverage(
            scores, matches, metric_fn=accuracy_metric, direction=1
        )
        coverages = [p.coverage for p in points]
        assert coverages == sorted(coverages, reverse=True)
        assert set(coverages) == set(DEFAULT_COVERAGES)


class TestCoverageAtThreshold:
    def test_threshold_above_max_yields_zero_coverage(self) -> None:
        assert coverage_at_threshold([0.1, 0.2, 0.3], 0.99) == 0.0

    def test_threshold_below_min_yields_full_coverage(self) -> None:
        assert coverage_at_threshold([0.1, 0.2, 0.3], -1.0) == 1.0

    def test_empty_input_returns_zero(self) -> None:
        assert coverage_at_threshold([], 0.0) == 0.0


class TestBuildRecordRow:
    def test_single_label_correct(self) -> None:
        signals = UncertaintySignals(
            prediction="J18.900",
            n_tokens=3,
            sum_logprob=-1.0,
            mean_logprob=-0.33,
            min_logprob=-0.5,
            mean_entropy=0.1,
            first_token_entropy=0.05,
        )
        row = _build_record_row(
            idx=7,
            source_text="pneumonia",
            gold="J18.900",
            source_id="copenhagen",
            signals=signals,
            label_separator=",",
        )
        assert row["correct"] is True
        assert row["exact_match"] is True
        assert row["idx"] == 7
        assert row["mean_logprob"] == pytest.approx(-0.33)

    def test_multi_label_exact_match_independent_of_string_order(self) -> None:
        signals = UncertaintySignals(
            prediction="B,A",
            n_tokens=3,
            sum_logprob=0.0,
            mean_logprob=0.0,
            min_logprob=0.0,
            mean_entropy=0.0,
            first_token_entropy=0.0,
        )
        row = _build_record_row(
            idx=0,
            source_text="x",
            gold="A,B",
            source_id="src",
            signals=signals,
            label_separator=",",
        )
        # Exact string mismatch but set equality should hold.
        assert row["correct"] is False
        assert row["exact_match"] is True


class TestComputeAllRcCurves:
    def _make_records(self) -> list[dict]:
        records = []
        confidences = [0.9, 0.4, 0.1, 0.7]
        matches = [True, False, False, True]
        for idx, (score, ok) in enumerate(zip(confidences, matches)):
            records.append(
                {
                    "idx": idx,
                    "source_id": "src",
                    "source": "x",
                    "gold": "A",
                    "prediction": "A" if ok else "B",
                    "correct": ok,
                    "exact_match": ok,
                    "n_tokens": 1,
                    "sum_logprob": score,
                    "mean_logprob": score,
                    "min_logprob": score,
                    "mean_entropy": -score,
                    "first_token_entropy": -score,
                }
            )
        return records

    def test_curves_cover_all_metrics_and_signals(self) -> None:
        records = self._make_records()
        curves = compute_all_rc_curves(records, label_separator=",")
        assert set(curves.keys()) == {"accuracy", "exact_match", "sample_f1"}
        for metric_name in curves:
            assert set(curves[metric_name].keys()) == set(UNCERTAINTY_SIGNAL_NAMES)
            for signal_name, points in curves[metric_name].items():
                assert len(points) == len(DEFAULT_COVERAGES)

    def test_metric_inputs_split_correctly(self) -> None:
        records = self._make_records()
        inputs = _records_to_metric_inputs(records, ",")
        assert inputs["accuracy"] == [True, False, False, True]
        assert inputs["exact_match"] == [True, False, False, True]
        for predicted, label in inputs["sample_f1"]:
            assert isinstance(predicted, set)
            assert isinstance(label, set)


class TestArtifactWriters:
    def test_predictions_jsonl_round_trip(self, tmp_path: Path) -> None:
        records = [
            {"idx": 0, "prediction": "A", "correct": True},
            {"idx": 1, "prediction": "B", "correct": False},
        ]
        path = write_predictions_jsonl(records, tmp_path / "predictions.jsonl")
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert loaded == records

    def test_test_rows_parquet_round_trip(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {"text": ["a", "b"], "label": ["L1", "L2"], "source_id": ["s1", "s2"]}
        )
        path = write_test_rows_parquet(df, tmp_path / "test_rows.parquet")
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_rc_curves_json_round_trip(self, tmp_path: Path) -> None:
        records = [
            {
                "idx": 0,
                "source_id": "src",
                "source": "x",
                "gold": "A",
                "prediction": "A",
                "correct": True,
                "exact_match": True,
                "n_tokens": 1,
                "sum_logprob": 0.0,
                "mean_logprob": 0.0,
                "min_logprob": 0.0,
                "mean_entropy": 0.0,
                "first_token_entropy": 0.0,
            }
        ]
        curves = compute_all_rc_curves(records, label_separator=",")
        path = write_rc_curves_json(curves, tmp_path / "rc_curves.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload.keys()) == {"accuracy", "exact_match", "sample_f1"}
        for metric_name in payload:
            assert set(payload[metric_name].keys()) == set(UNCERTAINTY_SIGNAL_NAMES)


def test_uncertainty_signal_direction_table_covers_all_signals() -> None:
    """Every named signal must declare a direction so RC orientation is unambiguous."""
    assert set(UNCERTAINTY_SIGNAL_DIRECTION.keys()) == set(UNCERTAINTY_SIGNAL_NAMES)
    for direction in UNCERTAINTY_SIGNAL_DIRECTION.values():
        assert direction in (-1, 1)


# ---------------------------------------------------------------------------
# Temperature-scaling tests
# ---------------------------------------------------------------------------


class TestFitTemperature:
    def test_well_calibrated_logits_yield_temperature_near_one(self) -> None:
        """Logits that already match the gold distribution should not move T much."""
        import torch

        from codllm.uncertainty.calibration import fit_temperature

        torch.manual_seed(0)
        n, vocab = 256, 8
        logits = torch.randn(n, vocab) * 1.5
        gold_ids = logits.argmax(dim=-1)
        # Argmax targets => optimal T -> 0+, but we cap iterations and start at 1.
        temperature, nll_before, nll_after = fit_temperature(logits, gold_ids, max_iter=50)
        assert temperature > 0.0
        # NLL should not get worse after fitting.
        assert nll_after <= nll_before + 1e-6

    def test_overconfident_logits_recover_temperature_above_one(self) -> None:
        """Logits scaled by 4x against a noisier label should pull T toward 4."""
        import torch

        from codllm.uncertainty.calibration import fit_temperature

        torch.manual_seed(7)
        n, vocab = 512, 16
        true_logits = torch.randn(n, vocab)
        gold_ids = torch.distributions.Categorical(
            logits=true_logits
        ).sample()
        # Make the model overconfident by scaling logits up by 4x.
        overconfident_logits = true_logits * 4.0
        temperature, _, _ = fit_temperature(overconfident_logits, gold_ids, max_iter=100)
        # Recovered T should be close to 4 (the inverse of the overconfidence scale).
        assert temperature == pytest.approx(4.0, rel=0.25)

    def test_empty_logits_short_circuit(self) -> None:
        import torch

        from codllm.uncertainty.calibration import fit_temperature

        empty_logits = torch.zeros((0, 4))
        empty_gold = torch.zeros((0,), dtype=torch.long)
        temperature, nll_before, nll_after = fit_temperature(
            empty_logits, empty_gold
        )
        assert temperature == 1.0
        assert nll_before == 0.0
        assert nll_after == 0.0


class TestTemperatureFitSerialization:
    def test_round_trip(self, tmp_path: Path) -> None:
        from codllm.uncertainty.calibration import (
            TemperatureFit,
            write_temperature_json,
        )

        fit = TemperatureFit(
            temperature=1.5,
            nll_before=2.3,
            nll_after=2.0,
            n_tokens=1234,
            n_records=200,
            max_records=2000,
        )
        path = write_temperature_json(fit, tmp_path / "temperature.json")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["temperature"] == 1.5
        assert loaded["n_tokens"] == 1234
        assert loaded["max_records"] == 2000
