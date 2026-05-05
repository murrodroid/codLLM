from io import StringIO
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


class _FakeRun:
    """Minimal W&B run replacement for artifact tests."""

    id = "run-1"
    name = "run"
    step = 7

    def __init__(self) -> None:
        self.artifacts: list[_FakeArtifact] = []

    def log_artifact(self, artifact: "_FakeArtifact", aliases: list[str]) -> None:
        """Record a logged artifact and aliases."""
        artifact.aliases = aliases
        self.artifacts.append(artifact)


class _FakeArtifactFile(StringIO):
    """StringIO that stores content on close."""

    def __init__(self, artifact: "_FakeArtifact", name: str) -> None:
        super().__init__()
        self.artifact = artifact
        self.name = name

    def close(self) -> None:
        self.artifact.files[self.name] = self.getvalue()
        super().close()


class _FakeArtifact:
    """Minimal W&B artifact replacement for visualization tests."""

    def __init__(self, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: dict[str, str] = {}
        self.aliases: list[str] = []

    def new_file(self, name: str, mode: str) -> _FakeArtifactFile:
        """Return a writable fake artifact file."""
        del mode
        return _FakeArtifactFile(self, name)


class _FakeArtifactWandb(_FakeWandb):
    """Fake W&B module with artifact support."""

    def __init__(self) -> None:
        super().__init__()
        self.run = _FakeRun()

    def Artifact(self, name: str, type: str, metadata: dict[str, Any]) -> _FakeArtifact:
        """Build a fake artifact."""
        return _FakeArtifact(name=name, type=type, metadata=metadata)


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


def test_metric_artifact_logger_logs_full_evaluation_artifact(monkeypatch) -> None:
    """Prediction artifacts should preserve full metrics and row-level outputs."""
    fake_wandb = _FakeArtifactWandb()
    monkeypatch.setattr(visualizations_module, "_active_wandb", lambda: fake_wandb)
    logger = MetricArtifactLogger(Config(label_separator=","))

    logger(
        metric_scope="test",
        input_strings=["cod: alpha", "cod: beta"],
        predictions=["A00.001", "B00.001"],
        labels=["A00.001", "A01.001"],
        metrics={"accuracy": 0.5, "seen_accuracy": 1.0},
    )

    artifact = fake_wandb.run.artifacts[0]
    assert artifact.type == "evaluation"
    assert artifact.metadata == {"scope": "test", "rows": 2}
    assert '"seen_accuracy": 1.0' in artifact.files["metrics.json"]
    assert '"prediction": "B00.001"' in artifact.files["predictions.jsonl"]
    assert artifact.aliases == ["test", "latest"]


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
    scalar_table = merged_payload["data/preparation_scalar_metrics"]
    assert ["balance", "enabled", "true"] in scalar_table.data
    assert ["balance", "strategy", "sqrt"] in scalar_table.data
    assert all(isinstance(row[2], str) for row in scalar_table.data)
