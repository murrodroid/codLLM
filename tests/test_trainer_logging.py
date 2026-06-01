"""Fast unit tests for trainer_logging pure helpers (no Trainer instantiation)."""

from __future__ import annotations

from types import SimpleNamespace

from codllm.trainer_logging import (
    _eval_dataset_from_evaluate_call,
    _metric_key_prefix_from_evaluate_call,
    _resolve_source_ids,
    rewrite_metric_key_for_stage,
)


class TestResolveSourceIds:
    def test_returns_none_for_missing_dataset(self) -> None:
        assert _resolve_source_ids(None) is None

    def test_returns_none_when_attribute_absent(self) -> None:
        dataset = SimpleNamespace()
        assert _resolve_source_ids(dataset) is None

    def test_returns_list_copy_when_present(self) -> None:
        original = ["a", "b"]
        dataset = SimpleNamespace(source_ids=original)

        result = _resolve_source_ids(dataset)

        assert result == ["a", "b"]
        assert result is not original


class TestEvalDatasetFromEvaluateCall:
    def test_prefers_kwarg_eval_dataset(self) -> None:
        chosen = _eval_dataset_from_evaluate_call(
            self_eval_dataset="default_ds",
            args=("from_args",),
            kwargs={"eval_dataset": "from_kwargs"},
        )
        assert chosen == "from_kwargs"

    def test_falls_back_to_first_positional_arg(self) -> None:
        chosen = _eval_dataset_from_evaluate_call(
            self_eval_dataset="default_ds",
            args=("positional_ds",),
            kwargs={},
        )
        assert chosen == "positional_ds"

    def test_uses_self_eval_dataset_when_no_arg(self) -> None:
        chosen = _eval_dataset_from_evaluate_call(
            self_eval_dataset="default_ds",
            args=(),
            kwargs={},
        )
        assert chosen == "default_ds"

    def test_explicit_none_falls_back_to_self(self) -> None:
        """Passing eval_dataset=None should still hit the trainer's default ds."""
        chosen = _eval_dataset_from_evaluate_call(
            self_eval_dataset="default_ds",
            args=(),
            kwargs={"eval_dataset": None},
        )
        assert chosen == "default_ds"


class TestMetricKeyPrefixFromEvaluateCall:
    def test_default_prefix_is_eval(self) -> None:
        assert _metric_key_prefix_from_evaluate_call((), {}) == "eval"

    def test_kwarg_overrides_default(self) -> None:
        assert (
            _metric_key_prefix_from_evaluate_call((), {"metric_key_prefix": "test"})
            == "test"
        )

    def test_third_positional_arg_used(self) -> None:
        assert (
            _metric_key_prefix_from_evaluate_call((None, None, "holdout_val"), {})
            == "holdout_val"
        )


class TestRewriteMetricKeyForStage:
    def test_eval_keys_routed_to_val(self) -> None:
        assert rewrite_metric_key_for_stage("eval_accuracy", "train") == "val/accuracy"

    def test_test_keys_routed_to_test(self) -> None:
        assert rewrite_metric_key_for_stage("test_accuracy", "train") == "test/accuracy"

    def test_per_source_eval_key_keeps_source_prefix(self) -> None:
        """Per-source eval metrics must remain under val/ namespace untouched."""
        rewritten = rewrite_metric_key_for_stage(
            "eval_source_copenhagen_accuracy", "train"
        )
        assert rewritten == "val/source_copenhagen_accuracy"

    def test_pretraining_stage_routes_to_pretraining_namespace(self) -> None:
        assert (
            rewrite_metric_key_for_stage("eval_macro_f1", "pretrain")
            == "pretraining/val/macro_f1"
        )
