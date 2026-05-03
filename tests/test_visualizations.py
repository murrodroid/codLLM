from typing import Any

import pandas as pd

from codllm.config import Config
from codllm.data import DataSplits
import codllm.training.visualizations as visualizations_module
from codllm.training.visualizations import MetricArtifactLogger, log_data_visualizations


class _FakeTable:
    """Minimal W&B Table replacement for visualization tests."""

    def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class _FakePlot:
    """Minimal W&B plot namespace replacement for visualization tests."""

    def bar(
        self,
        table: _FakeTable,
        x_column: str,
        y_column: str,
        title: str,
    ) -> dict[str, Any]:
        """Return a serializable marker for a requested bar chart."""
        return {
            "table": table,
            "x_column": x_column,
            "y_column": y_column,
            "title": title,
        }

    def histogram(
        self,
        table: _FakeTable,
        value_column: str,
        title: str,
    ) -> dict[str, Any]:
        """Return a serializable marker for a requested histogram."""
        return {
            "table": table,
            "value_column": value_column,
            "title": title,
        }


class _FakeWandb:
    """Minimal active W&B module replacement for visualization tests."""

    def __init__(self) -> None:
        self.run = object()
        self.plot = _FakePlot()
        self.logs: list[dict[str, Any]] = []

    def Table(self, columns: list[str], data: list[list[Any]]) -> _FakeTable:
        """Build a fake table."""
        return _FakeTable(columns=columns, data=data)

    def log(self, payload: dict[str, Any]) -> None:
        """Record a logged payload."""
        self.logs.append(payload)


def test_metric_artifact_logger_logs_chapter_block_error_tables(
    monkeypatch,
) -> None:
    """Prediction error tables should aggregate labels by chapter-block."""
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(visualizations_module, "_active_wandb", lambda: fake_wandb)
    logger = MetricArtifactLogger(Config(label_separator=","))

    logger(
        metric_scope="val",
        input_strings=["cod: alpha", "cod: beta"],
        predictions=["A00.001", "B00.001"],
        labels=["A00.001", "A01.001"],
    )

    merged_payload = {
        key: value for payload in fake_wandb.logs for key, value in payload.items()
    }
    pair_table = merged_payload["val/errors/chapter_block_pairs"]
    false_negative_table = merged_payload["val/errors/chapter_block_false_negatives"]
    false_positive_table = merged_payload["val/errors/chapter_block_false_positives"]

    assert pair_table.columns == [
        "true_chapter_blocks",
        "predicted_chapter_blocks",
        "count",
        "example_label",
        "example_prediction",
        "example_input",
    ]
    assert pair_table.data[0][:3] == ["A01", "B00", 1]
    assert false_negative_table.data[0] == ["A01", 1, 1, 1.0]
    assert false_positive_table.data[0] == ["B00", 1]


def test_log_data_visualizations_logs_core_distribution_tables(monkeypatch) -> None:
    """Data visualization logging should publish compact split and balance tables."""
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(visualizations_module, "_active_wandb", lambda: fake_wandb)
    cfg = Config(label_separator=",")
    splits = DataSplits(
        train=pd.DataFrame(
            {
                "source_id": ["real", "synthetic_multicod:real"],
                "text": ["cod: alpha | age: 1", "cod: beta | age: 2"],
                "label": ["A00.001", "A01.001,B00.001"],
            }
        ),
        val=pd.DataFrame(
            {
                "source_id": ["real"],
                "text": ["cod: gamma | age: 3"],
                "label": ["A00.002"],
            }
        ),
        test=pd.DataFrame(
            {
                "source_id": ["external"],
                "text": ["cod: delta | age: 4"],
                "label": ["C00.001"],
            }
        ),
    )
    balance_metrics = {
        "enabled": True,
        "strategy": "sqrt",
        "rows_before": 1,
        "rows_added": 1,
        "rows_after": 2,
        "base_perturbed_rows": 1,
        "label_distribution_before": {"A00": 1},
        "label_distribution_after": {"A00": 1, "A01": 1, "B00": 1},
    }

    log_data_visualizations(
        cfg=cfg,
        splits=splits,
        balance_metrics=balance_metrics,
    )

    merged_payload = {
        key: value for payload in fake_wandb.logs for key, value in payload.items()
    }
    assert "data/split_rows" in merged_payload
    assert "data/chapter_block_distribution" in merged_payload
    assert "data/balance_chapter_block_before_after" in merged_payload
    assert "data/seen_unseen_string_coverage" in merged_payload
    assert ["A01", 0, 1, 1] in merged_payload[
        "data/balance_chapter_block_before_after"
    ].data
