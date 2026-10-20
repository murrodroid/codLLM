"""CPU fixtures for paired resampling, metric identity, and frozen label coverage."""

import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codllm.config import Config as TrainingConfig
from codllm.evaluation.bootstrap import (
    ClusterCounts,
    Config,
    align_predictions,
    bootstrap_scores,
    group_indices,
    intervals,
)
from codllm.evaluation.campaign_analysis import comparison_pairs, read_selected
from codllm.evaluation.coverage import summarize_coverage
from codllm.evaluation.scoring import reference_macro_f1, score_slice


def predictions() -> pd.DataFrame:
    """Build linked, repeated and cross-source CODs with rare and multi-label targets."""
    return align_predictions(
        pd.DataFrame(
            {
                "row_uid": list("abcdef"),
                "source_id": ["one", "one", "one", "two", "two", "two"],
                "record_id": ["1", "1", "2", "3", "4", "5"],
                "cod_hash": ["x", "y", "z", "z", "w", "v"],
                "labels": [
                    ["A00.001"],
                    ["B00.001"],
                    ["A00.001", "B00.001"],
                    ["A00.001"],
                    ["C00.001"],
                    ["A00.001"],
                ],
                "predictions": [
                    ["A00.001"],
                    ["A00.001"],
                    ["B00.001"],
                    ["Z00.001"],
                    [],
                    ["A00.002"],
                ],
            }
        )
    )


def test_counts_reproduce_published_scores_and_weighted_resamples() -> None:
    """Compressed counts match direct scoring, including false-positive-only classes."""
    frame = predictions()
    groups = group_indices(frame)
    assert len(set(groups)) == 4
    counts = ClusterCounts.from_frame(frame, groups)
    for weights in (
        np.ones(4, dtype=int),
        np.array([0, 2, 1, 1]),
        np.array([1, 1, 2, 0]),
    ):
        indices = np.repeat(np.arange(len(frame)), weights[groups])
        resampled = frame.iloc[indices]
        targets = [set(value) for value in resampled["labels"]]
        predicted = [set(value) for value in resampled["predictions"]]
        direct = score_slice(predicted, targets)
        scores = counts.evaluate(weights)
        for index, key in (
            (1, "micro_f1"),
            (3, "exact_match"),
            (4, "sample_f1"),
            (5, "chapter_block_micro_f1"),
            (0, "macro_f1"),
        ):
            assert scores[index] == pytest.approx(direct[key])
        source_mean = np.mean(
            [
                reference_macro_f1(
                    [set(v) for v in part["predictions"]],
                    [set(v) for v in part["labels"]],
                )
                for _, part in resampled.groupby("source_id")
            ]
        )
        assert scores[2] == pytest.approx(source_mean)
    point = counts.evaluate(np.ones(4))
    assert point[0] == point[6]
    assert point[2] == point[7]


def test_fixed_support_keeps_missing_labels_and_archives_in_denominator() -> None:
    """An omitted rare code cannot silently improve the fixed-support macro average."""
    frame = predictions()
    counts = ClusterCounts.from_frame(frame, group_indices(frame))
    weights = np.array([0, 0, 1, 0])
    score = counts.evaluate(weights)
    assert score[6] <= score[0]
    assert score[7] <= score[2]
    assert counts.support_diagnostics()["source_label_pairs_in_one_group"] > 0


def test_pairing_rejects_changed_record_links_and_rows() -> None:
    """Equal target/COD values do not permit different source-record dependence graphs."""
    frame = predictions()
    changed = frame.copy()
    changed.loc[0, "record_id"] = "changed"
    with pytest.raises(ValueError, match="record_id"):
        align_predictions(changed, frame)
    with pytest.raises(ValueError, match="same rows"):
        align_predictions(frame.iloc[:-1], frame)
    with pytest.raises(ValueError, match="unique"):
        align_predictions(pd.concat([frame, frame.iloc[:1]]))


@pytest.mark.parametrize("weighting", ["bayesian", "multinomial"])
def test_paired_draws_are_order_invariant_and_reverse_sign(weighting: str) -> None:
    """Shared draws yield exact zero for identical models and symmetric reversed differences."""
    config = Config(replicates=40, weighting=weighting)
    frame = predictions()
    first, a, _ = bootstrap_scores(frame, config)
    second, b, _ = bootstrap_scores(align_predictions(frame.iloc[::-1]), config)
    assert np.array_equal(a, b)
    assert intervals(first - second, a - b, config)["macro_f1"]["ci95"] == [0, 0]
    changed = frame.copy()
    changed["predictions"] = changed["labels"]
    third, c, _ = bootstrap_scores(changed, config)
    forward = intervals(first - third, a - c, config)
    backward = intervals(third - first, c - a, config)
    assert forward["macro_f1"]["ci95"] == pytest.approx(
        [-v for v in reversed(backward["macro_f1"]["ci95"])]
    )
    with pytest.raises(ValueError, match="replicates"):
        bootstrap_scores(frame, replace(config, replicates=1))


def test_coverage_distinguishes_types_occurrences_rows_and_masterlist() -> None:
    """Historical zero-shot labels can be covered by masterlist exposure on mixed-label rows."""
    frame = pd.DataFrame(
        {"source_id": ["one", "one", "two"], "label": ["A,B", "A", "C"]}
    )
    result = summarize_coverage(
        frame, {"A"}, {"one": {"A"}}, {"B"}, {"A", "B"}, TrainingConfig()
    )
    assert result["distinct_code_coverage"] == 1 / 3
    assert result["target_coverage"] == 0.5
    assert result["rows_any_unseen_label"] == 2
    assert result["rows_no_labels_seen"] == 1
    assert result["historically_unseen_codes_in_masterlist"] == 1
    assert result["codes_absent_from_all_adaptation"] == 1
    assert result["targets_seen_in_same_source"] == 2


def test_comparison_inventory_covers_all_campaigns() -> None:
    """The complete factorial grid yields 112 recipe, 16 input and 16 patience pairs."""
    runs = []
    for input_name, patience, floor, synthesis, pretraining in itertools.product(
        ("cod", "metadata"), (10, 20), (0, 450), (0.3, 0.6), (4, 48)
    ):
        cell = f"floor-{floor}_synthetic-{synthesis}_pretrain-{pretraining}"
        runs.append(
            {
                "id": f"{input_name}-p{patience}/{cell}",
                "cell": cell,
                "input": input_name,
                "patience": patience,
                "recipe": {
                    "floor": floor,
                    "synthesis": synthesis,
                    "pretraining": pretraining,
                },
            }
        )
    pairs = comparison_pairs(runs)
    assert len(pairs) == 144
    assert sum(pair["kind"] == "metadata" for pair in pairs) == 16
    assert sum(pair["kind"] == "patience" for pair in pairs) == 16
    assert len({frozenset((pair["first"], pair["second"])) for pair in pairs}) == len(
        pairs
    )


def test_selected_artifact_digest_rejects_modified_predictions(tmp_path: Path) -> None:
    """File existence alone cannot certify selected prediction identity."""
    from codllm.evaluation.artifacts import write_json
    from codllm.evaluation.provenance import content_digest

    frame = predictions()
    records = frame.to_dict("records")
    payload = {"identity": {"scope": "val"}, "records": records}
    digest = content_digest(payload)
    frame.to_parquet(tmp_path / f"{digest}.parquet")
    write_json(
        tmp_path / "selected.json",
        {
            "identity": payload["identity"],
            "digest": digest,
            "predictions": f"{digest}.parquet",
            "metrics": {},
        },
    )
    read_selected(tmp_path / "selected.json")
    frame.loc[0, "source_id"] = "changed"
    frame.to_parquet(tmp_path / f"{digest}.parquet")
    with pytest.raises(ValueError, match="content changed"):
        read_selected(tmp_path / "selected.json")


def test_coverage_uses_verified_original_manifests(tmp_path: Path) -> None:
    """Coverage checks the reference fingerprint and original train/held-out label counts."""
    from types import SimpleNamespace

    from codllm.evaluation.artifacts import persist_training_contract
    from codllm.evaluation.coverage import label_coverage
    from codllm.evaluation.reference import build_reference

    cfg = TrainingConfig(output_dir=str(tmp_path), evaluation_protocol="cod")
    frame = pd.DataFrame(
        {
            "row_uid": list("abcd"),
            "record_id": list("abcd"),
            "source_id": ["one"] * 4,
            "cod_key": list("wxyz"),
            "cod_text": list("wxyz"),
            "text": list("wxyz"),
            "label": ["A", "A", "A,B", "B"],
            "language": ["en"] * 4,
            "language_candidates": ["en"] * 4,
        }
    )
    train = frame.iloc[:2]
    masterlist = frame.iloc[3:]
    reference = build_reference(train, train, masterlist, cfg)
    directory = persist_training_contract(
        cfg,
        SimpleNamespace(
            original_train=train, val=frame.iloc[2:3], test=frame.iloc[3:], holdout=None
        ),
        reference,
    )
    report = label_coverage(directory, cfg)
    assert report["partitions"]["test"]["all"]["historically_unseen_codes"] == 1
    assert (
        report["partitions"]["test"]["all"]["historically_unseen_codes_in_masterlist"]
        == 1
    )
    assert report["partitions"]["val"]["all"]["distinct_code_coverage"] == 0.5
