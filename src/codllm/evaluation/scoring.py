"""Support-aware source, language, novelty, and composition evaluation."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np

from codllm.evaluation.reference import EvaluationReference, cod_hash


def reference_macro_f1(predictions: list[set[str]], labels: list[set[str]]) -> float:
    """Average code F1 over the fixed reference-supported label universe of this slice."""
    true_counts: dict[str, int] = defaultdict(int)
    predicted_counts: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for predicted, target in zip(predictions, labels, strict=True):
        for code in target:
            true_counts[code] += 1
        for code in predicted:
            predicted_counts[code] += 1
        for code in predicted & target:
            hits[code] += 1
    return (
        float(
            np.mean(
                [
                    2 * hits[code] / (count + predicted_counts[code])
                    for code, count in true_counts.items()
                ]
            )
        )
        if true_counts
        else float("nan")
    )


def score_slice(
    predictions: list[set[str]], labels: list[set[str]]
) -> dict[str, float]:
    """Score full prediction sets, keeping false positives even on eligible-target slices."""
    from codllm.metrics import _prediction_metric_set

    metrics = _prediction_metric_set(
        predictions,
        labels,
        [p == y for p, y in zip(predictions, labels, strict=True)],
        multi_label=True,
    )
    if not labels:
        metrics = {key: float("nan") for key in metrics}
    metrics.update(
        macro_f1_ref=reference_macro_f1(predictions, labels),
        row_count=float(len(labels)),
        code_count=float(len(set().union(*labels))) if labels else 0.0,
        target_count=float(sum(map(len, labels))),
    )
    return metrics


def crosslingual_codes(
    codes: set[str],
    language: str,
    reference: EvaluationReference,
    *,
    strict_adaptation: bool = False,
) -> set[str]:
    """Identify codes taught only in other known languages under the declared exposure policy."""
    if not language or language in {"und", "mul"}:
        return set()
    eligible = set()
    for code in codes:
        historical = reference.label_languages.get(code, set())
        if not historical or historical & {language, "und", "mul", ""}:
            continue
        if strict_adaptation and reference.adaptation_languages.get(code, set()) & {
            language,
            "und",
            "mul",
            "",
        }:
            continue
        eligible.add(code)
    return eligible


def score_publication(
    predictions: list[set[str]],
    labels: list[set[str]],
    rows: list[dict[str, str]],
    reference: EvaluationReference,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute versioned metrics and auditable per-example eligibility records."""
    if not (len(predictions) == len(labels) == len(rows)):
        raise ValueError("Prediction, reference, and original-row alignment differs.")
    metrics = score_slice(predictions, labels)
    buckets: dict[str, list[int]] = defaultdict(list)
    transfer_targets: dict[str, list[tuple[int, set[str]]]] = defaultdict(list)
    records = []
    for index, (predicted, target, row) in enumerate(
        zip(predictions, labels, rows, strict=True)
    ):
        language = row["language"]
        code_key = cod_hash(row["cod_key"])
        seen = code_key in reference.historical_cods
        historical_transfer = crosslingual_codes(target, language, reference)
        strict_transfer = crosslingual_codes(
            target, language, reference, strict_adaptation=True
        )
        source_transfer = {
            code
            for code in target
            if reference.label_sources.get(code)
            and row["source_id"] not in reference.label_sources[code]
        }
        unseen_labels = {code for code in target if code not in reference.label_sources}
        source_key = re.sub(r"[^a-zA-Z0-9_]", "_", row["source_id"])
        row_buckets = [
            "seen_cod" if seen else "unseen_cod",
            "single_cod" if len(target) == 1 else "multi_cod",
            f"source_{source_key}",
            f"language_{language}",
        ]
        if code_key in reference.masterlist_cods:
            row_buckets.append("masterlist_seen_cod")
        if len(target) > 1 and not unseen_labels:
            row_buckets.append(
                "known_code_combination"
                if ",".join(sorted(target)) in reference.combinations
                else "novel_code_combination"
            )
        for name, eligible in (
            ("crosslingual", historical_transfer),
            ("strict_crosslingual", strict_transfer),
            ("source_transfer", source_transfer),
            ("historically_unseen_label", unseen_labels),
        ):
            if eligible:
                row_buckets.append(name)
                row_buckets.append(f"{name}_language_{language}")
                transfer_targets[name].append((index, eligible))
                row_buckets.append(
                    f"{name}_single_cod" if len(target) == 1 else f"{name}_multi_cod"
                )
                if not seen:
                    row_buckets.append(f"{name}_unseen_cod")
        for name, low, high in (
            ("frequency_zero", 0, 0),
            ("frequency_1_9", 1, 9),
            ("frequency_10_99", 10, 99),
            ("frequency_100_plus", 100, float("inf")),
        ):
            if any(
                low <= reference.label_counts.get(code, 0) <= high for code in target
            ):
                row_buckets.append(name)
        for name in row_buckets:
            buckets[name].append(index)
        records.append(
            {
                "row_uid": row["row_uid"],
                "source_id": row["source_id"],
                "record_id": row["record_id"],
                "language": language,
                "cod_hash": code_key,
                "predictions": sorted(predicted),
                "labels": sorted(target),
                "seen_cod": seen,
                "adaptation_seen_cod": code_key in reference.adapted_cods,
                "masterlist_seen_cod": code_key in reference.masterlist_cods,
                "crosslingual_targets": sorted(historical_transfer),
                "strict_crosslingual_targets": sorted(strict_transfer),
                "source_transfer_targets": sorted(source_transfer),
                "historically_unseen_targets": sorted(unseen_labels),
                "masterlist_taught_targets": sorted(
                    target & reference.masterlist_labels
                ),
            }
        )
    for name in (
        "seen_cod",
        "unseen_cod",
        "crosslingual",
        "strict_crosslingual",
        "source_transfer",
        "historically_unseen_label",
    ):
        buckets.setdefault(name, [])
    for name, indices in buckets.items():
        metrics.update(
            {
                f"{name}_{key}": value
                for key, value in score_slice(
                    [predictions[i] for i in indices], [labels[i] for i in indices]
                ).items()
            }
        )
    for name in (
        "crosslingual",
        "strict_crosslingual",
        "source_transfer",
        "historically_unseen_label",
    ):
        targets = transfer_targets[name]
        occurrences = sum(len(codes) for _, codes in targets)
        hit_count = sum(len(codes & predictions[index]) for index, codes in targets)
        pair_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for index, codes in targets:
            for code in codes:
                pair = pair_counts[(code, rows[index]["language"])]
                pair[0] += int(code in predictions[index])
                pair[1] += 1
        metrics[f"{name}_eligible_target_count"] = float(occurrences)
        metrics[f"{name}_eligible_code_count"] = float(
            len({code for _, codes in targets for code in codes})
        )
        metrics[f"{name}_eligible_recall"] = (
            hit_count / occurrences if occurrences else float("nan")
        )
        metrics[f"{name}_eligible_macro_recall"] = (
            float(np.mean([hit / count for hit, count in pair_counts.values()]))
            if pair_counts
            else float("nan")
        )
        metrics[f"{name}_eligible_code_language_pairs"] = float(len(pair_counts))
    source_scores = [
        value
        for key, value in metrics.items()
        if key.startswith("source_")
        and key.endswith("_macro_f1_ref")
        and not key.startswith("source_transfer")
    ]
    metrics["source_mean_macro_f1_ref"] = (
        float(np.mean(source_scores)) if source_scores else float("nan")
    )
    metrics["source_worst_macro_f1_ref"] = (
        float(min(source_scores)) if source_scores else float("nan")
    )
    metrics["resolved_language_row_count"] = float(
        sum(row["language"] not in {"und", "mul", ""} for row in rows)
    )
    metrics["invalid_prediction_count"] = float(
        sum(
            not re.fullmatch(r"[A-Z]\d{2}\.\d{3}", code)
            for predicted in predictions
            for code in predicted
        )
    )
    return {f"pub_v1_{key}": value for key, value in metrics.items()}, records
