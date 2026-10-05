"""Synthetic CPU tests for publication checkpoint-window consistency calculations."""

from copy import deepcopy

import pytest

from experiments.checkpoint_consistency import analyze, analyze_run, summarize


def fixture_run() -> dict:
    """Build an isolated synthetic child window with a known linear trajectory."""
    return {
        "complete": True, "best_epoch": 20, "anchor_epoch": 10, "history_source": "child",
        "best_step": 200, "max_steps": 1200, "epochs_limit": 120, "best_metric": 0.20,
        "selected_metrics": {"pub_v1_macro_f1": 0.20},
        "contract": {"manifests": {"val": {"digest": "same", "rows": 12}}},
        "rows": [{"epoch": epoch, "step": epoch * 10, "metrics": {"macro_f1": epoch / 100}}
                 for epoch in range(11, 21)],
    }


def test_summary_definitions() -> None:
    """Check flat, increasing, endpoint-spike, and loss-direction examples."""
    flat = summarize([0.8] * 10)
    assert flat["sd"] == flat["slope_per_epoch"] == flat["endpoint_uplift"] == 0
    linear = summarize(list(range(10)))
    assert linear["mean"] == linear["median"] == 4.5
    assert linear["variance"] == 8.25
    assert linear["slope_per_epoch"] == 1
    assert linear["endpoint_uplift"] == 5
    spike = summarize([0.8] * 9 + [0.9])
    assert spike["endpoint_uplift"] == pytest.approx(0.1)
    assert summarize(list(range(10)), lower_is_better=True)["endpoint_uplift"] == -5


def test_window_and_missing_metric() -> None:
    """Keep one shared anchor and explicitly mark unavailable metrics."""
    run = fixture_run()
    run["rows"].append(deepcopy(run["rows"][-1]))
    result = analyze_run(run, ["macro_f1", "micro_f1"])
    assert result["window"] == [11, 20]
    assert len(result["rows"]) == 10
    assert result["summaries"]["macro_f1"]["restored_minus_endpoint"] == 0
    assert result["summaries"]["micro_f1"]["status"] == "unavailable"


@pytest.mark.parametrize("problem", ["missing", "conflict", "partial", "step", "fork", "restored"])
def test_invalid_windows(problem: str) -> None:
    """Refuse incomplete, spliced, duplicate-conflicting, or non-training observations."""
    run = fixture_run()
    if problem == "missing":
        run["rows"].pop(0)
    elif problem == "conflict":
        extra = deepcopy(run["rows"][-1])
        extra["metrics"]["macro_f1"] = 0.99
        run["rows"].append(extra)
    elif problem == "partial":
        run["rows"][0]["epoch"] = 11.5
    elif problem == "step":
        run["rows"][0]["step"] += 1
    elif problem == "fork":
        run["anchor_epoch"] = 12
    else:
        run["rows"].append({"epoch": 40, "step": 400, "metrics": {"macro_f1": 0.20}})
    with pytest.raises(ValueError):
        analyze_run(run, ["macro_f1"])


def test_inherited_and_validation_identity() -> None:
    """Allow explicitly inherited windows and reject mixed validation populations."""
    run = fixture_run()
    run.update(history_source="inherited_parent", anchor_epoch=20)
    assert analyze_run(run, ["macro_f1"])["history_source"] == "inherited_parent"
    other = deepcopy(run)
    other["contract"]["manifests"]["val"]["digest"] = "different"
    with pytest.raises(ValueError):
        analyze({"runs": [run, other], "metric_keys": ["macro_f1"]})
