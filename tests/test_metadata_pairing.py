"""CPU-only checks for metadata partition comparison against frozen historical inputs."""

import json
from pathlib import Path

import pandas as pd
import pytest

from codllm.config import Config
from codllm.data import DataHandler
from codllm.data.splits import DataSplits
from codllm.evaluation.artifacts import write_json, write_parquet
from codllm.evaluation.metadata_pairing import audit_metadata_pairing, compare_partition
from codllm.evaluation.provenance import content_digest


def frames() -> tuple[pd.DataFrame, pd.DataFrame, Config]:
    """Return tiny paired input tables with identical identities and differing metadata."""
    cfg = Config(training_input=["cod", "age", "sex"])
    reference = pd.DataFrame(
        {
            "row_uid": ["a", "b"],
            "source_id": ["archive", "archive"],
            "record_id": ["1", "2"],
            "cod_text": ["fever", "cough"],
            "cod_key": ["fever", "cough"],
            "language": ["en", "en"],
            "language_candidates": ["en", "en"],
            "text": ["fever", "cough"],
            "label": ["A00,B00", "C00"],
        }
    )
    actual = reference.copy()
    actual["text"] = ["cod: fever | age: unknown | sex: f", "cod: cough | age: 70 | sex: m"]
    actual.loc[0, "label"] = "B00,A00"
    return actual, reference, cfg


def test_partition_matches_without_using_demographics() -> None:
    """Accept metadata changes and label permutation without dropping missing metadata."""
    actual, reference, cfg = frames()
    assert compare_partition(actual, reference, cfg)["passed"]


@pytest.mark.parametrize("column", ["row_uid", "label", "cod_text", "text", "language"])
def test_partition_rejects_changes(column: str) -> None:
    """Reject identity, target, input-COD, or provenance drift."""
    actual, reference, cfg = frames()
    actual.loc[0, column] = "different"
    assert not compare_partition(actual, reference, cfg)["passed"]


def test_partition_rejects_order_and_membership() -> None:
    """Same counts alone do not establish a paired partition."""
    actual, reference, cfg = frames()
    assert not compare_partition(actual.iloc[::-1], reference, cfg)["passed"]
    assert not compare_partition(actual.iloc[:1], reference, cfg)["passed"]


@pytest.mark.parametrize("changed", [False, True])
def test_audit_compares_frozen_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: bool) -> None:
    """Exercise checksum-verified manifests and a score-blind saved result without building datasets."""
    actual, reference, cfg = frames()
    cfg.publication_eval_enabled = True
    cfg.evaluation_protocol = "cod"
    cfg.masterlist_inject_enabled = False
    parts = {}
    manifests = {}
    root = tmp_path / "reference"
    for name in ("original_train", "val", "test"):
        candidate = actual.copy()
        frozen = reference.copy()
        for column in ("row_uid", "record_id", "cod_key"):
            candidate[column] += name
            frozen[column] += name
        parts[name] = candidate
        if changed and name == "val":
            frozen.loc[0, "label"] = "Z99"
        digest = content_digest(frozen.fillna("").astype(str).to_dict("list"))
        manifests[name] = {"rows": len(frozen), "digest": digest}
        write_parquet(root / "manifests" / f"{name}-{digest[:16]}.parquet", frozen)
    write_json(root / "contract.json", {"manifests": manifests})
    splits = DataSplits(train=parts["original_train"], **parts)
    monkeypatch.setattr(DataHandler, "ensure_processed", lambda self: actual)
    monkeypatch.setattr(DataHandler, "prepare_original_splits", lambda self, data: (splits, []))
    output = tmp_path / "audit.json"
    report = audit_metadata_pairing([cfg], root, output)
    assert report["status"] == ("mismatch" if changed else "matched")
    assert json.loads(output.read_text()) == report
    assert "fever" not in output.read_text()


def test_metadata_augmentation_cache_version_is_isolated() -> None:
    """Invalidate legacy metadata augmentation caches without changing COD-only cache metadata."""
    plain = DataHandler(Config(training_input=["cod"]))._build_prepared_splits_metadata()
    metadata = DataHandler(Config(training_input=["cod", "age", "sex"]))._build_prepared_splits_metadata()
    assert "metadata_cod_perturbation_version" not in plain["split_config"]
    assert metadata["split_config"]["metadata_cod_perturbation_version"] == 2
