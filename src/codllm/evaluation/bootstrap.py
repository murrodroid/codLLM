"""Paired connected-group bootstrap of frozen predictions without model execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from codllm.evaluation.splitting import linked_groups


@dataclass(frozen=True)
class Config:
    """Own offline bootstrap settings independently of training configuration."""

    replicates: int = 10000
    seed: int = 777
    confidence: float = 0.95
    weighting: str = "bayesian"
    metrics: tuple[str, ...] = (
        "macro_f1",
        "micro_f1",
        "source_mean_macro_f1_ref",
        "exact_match",
        "sample_f1",
        "block_micro_f1",
        "macro_f1_fixed_support",
        "source_mean_macro_f1_ref_fixed_support",
    )

    def validate(self) -> None:
        """Reject settings that cannot define a percentile interval."""
        if self.replicates < 2 or not 0 < self.confidence < 1:
            raise ValueError(
                "Require at least two replicates and confidence between zero and one."
            )
        if self.weighting not in {"bayesian", "multinomial"}:
            raise ValueError("Bootstrap weighting must be bayesian or multinomial.")


def align_predictions(
    frame: pd.DataFrame, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Align unique row identities and reject inconsistent labels or dependence units."""
    required = {
        "row_uid",
        "source_id",
        "record_id",
        "cod_hash",
        "labels",
        "predictions",
    }
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(frame))}")
    if (
        frame["row_uid"].isna().any()
        or frame["row_uid"].duplicated().any()
        or frame.empty
    ):
        raise ValueError("Predictions require nonempty, unique row identities.")
    frame = frame.sort_values("row_uid").reset_index(drop=True).copy()
    for column in ("source_id", "record_id", "cod_hash"):
        if frame[column].isna().any():
            raise ValueError(f"Null {column} cannot define paired resampling units.")
    for column in ("labels", "predictions"):
        frame[column] = frame[column].map(lambda values: tuple(sorted(set(values))))
    if reference is not None:
        if frame["row_uid"].tolist() != reference["row_uid"].tolist():
            raise ValueError("Bootstrap requires predictions on exactly the same rows.")
        for column in ("labels", "source_id", "record_id", "cod_hash"):
            if frame[column].tolist() != reference[column].tolist():
                raise ValueError(f"Paired {column} identities differ.")
    return frame


def group_indices(frame: pd.DataFrame) -> np.ndarray:
    """Number connected global COD/source-record components in deterministic row order."""
    groups = linked_groups(frame.rename(columns={"cod_hash": "cod_key"}), by_cod=True)
    return np.unique(groups, return_inverse=True)[1]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with the published zero-division convention."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


@dataclass
class ClusterCounts:
    """Compress predictions into sufficient counts indexed by connected evaluation group."""

    groups: np.ndarray
    features: np.ndarray
    counts: np.ndarray
    row_counts: np.ndarray
    sources: int
    labels: int
    pooled_support: np.ndarray
    source_support: np.ndarray
    target_group_counts: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, groups: np.ndarray) -> ClusterCounts:
        """Accumulate sparse group/source/code and row-level metrics once per model."""
        sources = {value: i for i, value in enumerate(sorted(set(frame["source_id"])))}
        codes = sorted(
            {code for values in frame["labels"] for code in values}
            | {code for values in frame["predictions"] for code in values}
        )
        if not codes:
            raise ValueError("At least one label or prediction code is required.")
        labels = {code: i for i, code in enumerate(codes)}
        width = len(sources) * len(labels)
        coordinates, counts = [], []
        row_counts = np.zeros((int(groups.max()) + 1, 6), dtype=float)
        for group, (source, truth, prediction) in zip(
            groups,
            frame[["source_id", "labels", "predictions"]].itertuples(
                index=False, name=None
            ),
            strict=True,
        ):
            target, predicted = set(truth), set(prediction)
            for code in sorted(target | predicted):
                coordinates.append(
                    int(group) * width + sources[source] * len(labels) + labels[code]
                )
                counts.append(
                    (code in target, code in predicted, code in target & predicted)
                )
            block_target, block_predicted = (
                {code[:3] for code in target},
                {code[:3] for code in predicted},
            )
            denominator = len(target) + len(predicted)
            row_counts[group] += (
                1,
                target == predicted,
                2 * len(target & predicted) / denominator if denominator else 1,
                len(block_target),
                len(block_predicted),
                len(block_target & block_predicted),
            )
        unique, inverse = np.unique(coordinates, return_inverse=True)
        values = np.asarray(counts, dtype=float)
        compressed = np.array(
            [np.bincount(inverse, weights=values[:, i]) for i in range(3)]
        )
        features = unique % width
        totals = np.array(
            [
                np.bincount(features, weights=values, minlength=width)
                for values in compressed
            ]
        )
        targets, predicted, _ = totals.reshape(3, len(sources), len(labels))
        target_groups = np.bincount(
            features, weights=compressed[0] > 0, minlength=width
        )
        return cls(
            unique // width,
            features,
            compressed,
            row_counts,
            len(sources),
            len(labels),
            (targets + predicted).sum(axis=0) > 0,
            targets > 0,
            target_groups.reshape(targets.shape),
        )

    def evaluate(self, weights: np.ndarray) -> np.ndarray:
        """Recompute published metrics and a fixed-label-support sensitivity analysis."""
        weighted = weights[self.groups]
        totals = np.array(
            [
                np.bincount(
                    self.features,
                    weights=values * weighted,
                    minlength=self.sources * self.labels,
                )
                for values in self.counts
            ]
        ).reshape(3, self.sources, self.labels)
        truth, predicted, hits = totals
        pooled_truth, pooled_predicted, pooled_hits = totals.sum(axis=1)
        pooled_f1 = _ratio(2 * pooled_hits, pooled_truth + pooled_predicted)
        source_f1 = _ratio(2 * hits, truth + predicted)
        source_fixed = _ratio(
            (source_f1 * self.source_support).sum(axis=1),
            self.source_support.sum(axis=1),
        )
        observed_sources = truth.sum(axis=1) > 0
        source_dynamic = _ratio(
            (source_f1 * (truth > 0)).sum(axis=1), (truth > 0).sum(axis=1)
        )
        rows, exact, sample, block_truth, block_prediction, block_hits = (
            weights @ self.row_counts
        )
        return np.array(
            [
                pooled_f1[(pooled_truth + pooled_predicted) > 0].mean(),
                float(
                    _ratio(
                        np.asarray(2 * pooled_hits.sum()),
                        np.asarray(pooled_truth.sum() + pooled_predicted.sum()),
                    )
                ),
                source_dynamic[observed_sources].mean(),
                exact / rows,
                sample / rows,
                2 * block_hits / (block_truth + block_prediction)
                if block_truth + block_prediction
                else 0,
                pooled_f1[self.pooled_support].mean(),
                source_fixed.mean(),
            ]
        )

    def support_diagnostics(self) -> dict[str, Any]:
        """Report rare-code dependence support without exposing row identities or text."""
        active = self.target_group_counts[self.source_support]
        return {
            "pooled_macro_labels": int(self.pooled_support.sum()),
            "reference_labels_by_source": self.source_support.sum(axis=1).tolist(),
            "source_label_pairs": len(active),
            "source_label_pairs_in_one_group": int((active == 1).sum()),
            "source_label_pairs_in_fewer_than_five_groups": int((active < 5).sum()),
        }


def bootstrap_scores(
    frame: pd.DataFrame, config: Config
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate reproducible draws shared by every model evaluated on the same aligned rows."""
    config.validate()
    groups = group_indices(frame)
    group_count = int(groups.max()) + 1
    if group_count < 2:
        raise ValueError("At least two connected COD groups are required.")
    counts = ClusterCounts.from_frame(frame, groups)
    point = counts.evaluate(np.ones(group_count))
    random = np.random.default_rng(config.seed)
    samples = np.empty((config.replicates, len(config.metrics)))
    for index in range(config.replicates):
        if config.weighting == "bayesian":
            weights = random.exponential(size=group_count)
            weights *= group_count / weights.sum()
        else:
            weights = np.bincount(
                random.integers(group_count, size=group_count), minlength=group_count
            )
        samples[index] = counts.evaluate(weights)
    return (
        point,
        samples,
        {
            "groups": group_count,
            "rows": len(frame),
            "sources": sorted(set(frame["source_id"])),
            **counts.support_diagnostics(),
        },
    )


def intervals(
    point: np.ndarray, samples: np.ndarray, config: Config
) -> dict[str, dict[str, Any]]:
    """Summarize nominal percentile intervals for model scores or paired differences."""
    tail = (1 - config.confidence) / 2
    bounds = np.quantile(samples, [tail, 1 - tail], axis=0)
    return {
        key: {
            "estimate": float(point[i]),
            "ci95": bounds[:, i].tolist(),
            "bootstrap_mean": float(samples[:, i].mean()),
        }
        for i, key in enumerate(config.metrics)
    }
