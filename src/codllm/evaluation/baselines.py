"""Training-only lookup, character-linear, and masterlist-retrieval baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from codllm.config import Config
from codllm.data import DataHandler
from codllm.data.splits import DataSplits
from codllm.evaluation.artifacts import (
    PublicationReporter,
    load_manifest,
    persist_training_contract,
    write_json,
)
from codllm.evaluation.context import evaluation_rows
from codllm.evaluation.provenance import evaluation_records, load_external_table
from codllm.evaluation.reference import build_reference
from codllm.evaluation.scoring import score_publication
from codllm.evaluation.splitting import validate_group_integrity
from codllm.evaluation.workflow import validate_publication_config
from codllm.metrics import reset_metric_artifact_scope, set_metric_artifact_scope
from codllm.run_directory import prepare_run_output_dir


def labels_for(frame: pd.DataFrame, cfg: Config) -> list[set[str]]:
    """Read unordered target-code sets from a canonical dataframe."""
    return [
        {code.strip() for code in str(value).split(cfg.label_separator) if code.strip()}
        for value in frame[cfg.dataset_label_column]
    ]


def run_baseline(cfg: Config) -> dict[str, Any]:
    """Fit one baseline, choose any threshold on internal validation, and export selected predictions."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    validate_publication_config(cfg)
    if cfg.publication_baseline not in {"lookup", "linear", "retrieval"}:
        raise ValueError("publication_baseline must be lookup, linear, or retrieval.")
    prepare_run_output_dir(cfg)
    handler = DataHandler(cfg)
    if cfg.publication_gate == "final":
        if not cfg.evaluation_reference_dir:
            raise ValueError(
                "Final baselines require the frozen original run's CODLLM_EVALUATION_REFERENCE_DIR."
            )
        directory = Path(cfg.evaluation_reference_dir)
        original_train = load_manifest(directory, "original_train")
        splits = DataSplits(
            train=original_train,
            original_train=original_train,
            val=load_manifest(directory, "val"),
            test=load_manifest(directory, "test"),
        )
    else:
        splits = handler.get_splits()
    validate_group_integrity(splits, cfg)
    original = splits.original_train
    if original is None or splits.val.empty:
        raise ValueError(
            "Baselines require original training and validation manifests."
        )
    targets = labels_for(original, cfg)
    masterlist = None
    vectorizer = None
    estimator = None
    encoder = None
    training_matrix = None
    lookup: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    global_counts: Counter[tuple[str, ...]] = Counter(
        tuple(sorted(value)) for value in targets
    )
    fallback = set(min(global_counts, key=lambda value: (-global_counts[value], value)))
    if cfg.publication_baseline == "lookup":
        for key, labels in zip(original["cod_key"], targets, strict=True):
            lookup[key][tuple(sorted(labels))] += 1
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=cfg.baseline_max_features,
            dtype=np.float32,
        )
        if cfg.publication_baseline == "linear":
            training_matrix = vectorizer.fit_transform(
                original[cfg.dataset_text_column]
            )
            encoder = MultiLabelBinarizer(sparse_output=True)
            label_matrix = encoder.fit_transform(targets)
            estimator = OneVsRestClassifier(
                SGDClassifier(
                    loss="log_loss",
                    alpha=cfg.baseline_alpha,
                    max_iter=cfg.baseline_max_iter,
                    tol=1e-3,
                    random_state=cfg.seed,
                ),
                n_jobs=1,
            )
            estimator.fit(training_matrix, label_matrix)
        else:
            masterlist = handler._load_masterlist_source(
                "masterlist_labels",
                cfg.label_harmonization_masterlist_path,
                cfg.label_harmonization_masterlist_sheet_name,
            )
            training_matrix = vectorizer.fit_transform(
                masterlist[cfg.dataset_text_column]
            )
    reference = build_reference(original, original, masterlist, cfg)
    persist_training_contract(cfg, splits, reference)

    def predict(frame: pd.DataFrame, threshold: float) -> list[set[str]]:
        """Make bounded-memory predictions; no evaluation targets influence this function."""
        if cfg.publication_baseline == "lookup":
            return [
                set(min(lookup[key], key=lambda value: (-lookup[key][value], value)))
                if key in lookup
                else fallback.copy()
                for key in frame["cod_key"]
            ]
        predictions = []
        catalogue_targets = (
            labels_for(masterlist, cfg) if masterlist is not None else None
        )
        for start in range(0, len(frame), cfg.per_device_eval_batch_size):
            encoded = vectorizer.transform(
                frame[cfg.dataset_text_column].iloc[
                    start : start + cfg.per_device_eval_batch_size
                ]
            )
            if estimator is not None:
                scores = estimator.predict_proba(encoded)
                for row in scores:
                    indices = np.argsort(-row)[: cfg.max_label_count]
                    chosen = [
                        int(index) for index in indices if row[index] >= threshold
                    ]
                    predictions.append(set(encoder.classes_[chosen]))
            else:
                similarity = encoded @ training_matrix.T
                for row in similarity:
                    if not row.nnz:
                        predictions.append(set())
                    else:
                        best = int(row.indices[np.argmax(row.data)])
                        predictions.append(catalogue_targets[best].copy())
        return predictions

    thresholds = (
        cfg.baseline_thresholds if cfg.publication_baseline == "linear" else [0.5]
    )
    validation_labels = labels_for(splits.val, cfg)
    validation_rows = evaluation_records(splits.val, cfg)
    best_threshold = thresholds[0]
    best_score = -float("inf")
    for threshold in thresholds:
        metrics, _ = score_publication(
            predict(splits.val, threshold),
            validation_labels,
            validation_rows,
            reference,
        )
        score = metrics["pub_v1_source_mean_macro_f1_ref"]
        if score > best_score:
            best_score, best_threshold = score, threshold
    reporter = PublicationReporter(cfg, reference)
    reporter.exporting = True
    reporter.trainer = SimpleNamespace(
        state=SimpleNamespace(best_model_checkpoint=cfg.publication_baseline)
    )
    results = {}
    external = None
    if cfg.evaluation_data_path:
        if cfg.publication_gate != "final":
            raise ValueError(
                "External baseline evaluation requires a reviewed final gate."
            )
        external = load_external_table(cfg)
    for scope, data in (
        ("val", splits.val),
        ("holdout/full", splits.holdout),
        ("test", splits.test if cfg.final_test_eval_enabled else None),
        ("external", external),
    ):
        if data is None or data.empty:
            continue
        rows_token = evaluation_rows.set(evaluation_records(data, cfg))
        scope_token = set_metric_artifact_scope(scope)
        try:
            results[scope] = reporter(
                predict(data, best_threshold), labels_for(data, cfg)
            )
        finally:
            evaluation_rows.reset(rows_token)
            reset_metric_artifact_scope(scope_token)
    write_json(
        Path(cfg.output_dir) / "publication" / "baseline.json",
        {
            "method": cfg.publication_baseline,
            "selected_threshold": best_threshold,
            "masterlist_access": masterlist is not None,
            "historical_training_rows": len(original),
            "metrics": results,
        },
    )
    import joblib

    bundle = Path(cfg.output_dir) / "publication" / "baseline.joblib"
    joblib.dump(
        {
            "method": cfg.publication_baseline,
            "vectorizer": vectorizer,
            "estimator": estimator,
            "encoder": encoder,
            "lookup": dict(lookup),
            "fallback": fallback,
            "catalogue_matrix": training_matrix if masterlist is not None else None,
            "catalogue_labels": labels_for(masterlist, cfg)
            if masterlist is not None
            else None,
            "threshold": best_threshold,
        },
        bundle,
    )
    bundle.chmod(0o600)
    from codllm.evaluation.telemetry import log_completed_evaluation

    log_completed_evaluation(cfg, results, {"baseline": cfg.publication_baseline})
    return results
