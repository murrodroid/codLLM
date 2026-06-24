"""RQ1 baseline: TF-IDF + one-vs-rest LogisticRegression.

The classical machine-learning reference. Multi-label via OneVsRestClassifier
wrapping LogisticRegression, top-3 codes per record by predicted probability.

Reports macro-F1, micro-F1, sample-F1, per-code-level metrics, per-source slice.
Evaluated on the EXACT same 76,418-record test split as flan-t5 and the agentic
baseline (data_seed=333, max_label_count=3).

Usage:
    uv run python -m experiments.baselines.tfidf_baseline \\
        --max-features 200000 --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from experiments.baselines.common import (
    BASELINE_RESULTS_DIR,
    BaselineRecord,
    compute_baseline_metrics,
    extract_cod,
    load_train_and_test,
    per_record_prf1,
    save_snapshot,
    split_gold,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("tfidf_baseline")


def _parse_y_codes(y_codes_field, label_field: str) -> list[str]:
    """Pull a code list out of the row. The processed parquet has y_codes as a
    list-like (sometimes ndarray) and label as a comma-joined string."""
    if y_codes_field is not None:
        try:
            seq = list(y_codes_field)
            if seq:
                return [str(c).strip() for c in seq if str(c).strip()]
        except TypeError:
            pass
    return split_gold(label_field)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-features", type=int, default=200_000,
                   help="Max TF-IDF vocabulary size (default 200,000).")
    p.add_argument("--min-df", type=int, default=2,
                   help="Min document frequency for TF-IDF features (default 2).")
    p.add_argument("--ngram-max", type=int, default=2,
                   help="Upper end of TF-IDF n-gram range (default 2 = unigrams+bigrams).")
    p.add_argument("--min-code-count", type=int, default=20,
                   help="Only train classifiers for codes seen >= this many times in train. "
                        "Codes below the threshold cannot be predicted. Default 20 cuts the "
                        "label space from ~3,400 to ~700 trainable classes, keeping training "
                        "tractable on CPU.")
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of top codes to emit per record (default 3 to match max_label_count).")
    p.add_argument("--C", type=float, default=1.0,
                   help="Classifier C parameter (default 1.0).")
    p.add_argument("--max-iter", type=int, default=200,
                   help="Classifier max_iter (default 200).")
    p.add_argument("--classifier", choices=["linearsvc", "logreg"], default="linearsvc",
                   help="Per-class binary classifier (default linearsvc -- ~5x faster than "
                        "logreg on this corpus and similar ranking quality).")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Joblib parallel jobs for OneVsRest training (default -1 = all cores).")
    p.add_argument("--label", default="tfidf_ovr_logreg",
                   help="Result file label.")
    args = p.parse_args()

    label = args.label
    results_jsonl = BASELINE_RESULTS_DIR / f"{label}.predictions.jsonl"
    snapshot_path = BASELINE_RESULTS_DIR / f"{label}.results.json"

    logger.info("Loading train + test splits ...")
    t0 = time.time()
    train_df, test_df = load_train_and_test()
    logger.info("Loaded in %.1fs. train=%d  test=%d", time.time() - t0, len(train_df), len(test_df))

    text_col = "text"
    label_col = "label"
    y_codes_col = "y_codes"

    # ---- Extract bare CoD text and gold code sets ----------------------------
    train_texts = [extract_cod(t) for t in train_df[text_col]]
    train_labels = [_parse_y_codes(yc, lab) for yc, lab in zip(train_df[y_codes_col], train_df[label_col])]
    test_texts = [extract_cod(t) for t in test_df[text_col]]
    test_gold_str = list(test_df[label_col].astype(str))
    test_gold_codes = [_parse_y_codes(yc, lab) for yc, lab in zip(test_df[y_codes_col], test_df[label_col])]
    test_source_ids = list(test_df["source_id"].astype(str))
    test_parquet_idx = list(test_df.index.astype(int))

    # ---- Filter codes by frequency ------------------------------------------
    from collections import Counter
    code_counts = Counter()
    for codes in train_labels:
        for c in codes:
            code_counts[c] += 1
    kept_codes = [c for c, n in code_counts.items() if n >= args.min_code_count]
    logger.info(
        "Code vocabulary: %d unique in train, %d after min_code_count=%d filter "
        "(removed %d rare codes)",
        len(code_counts), len(kept_codes), args.min_code_count,
        len(code_counts) - len(kept_codes),
    )

    # Multi-label binarisation
    mlb = MultiLabelBinarizer(classes=sorted(kept_codes))
    Y_train = mlb.fit_transform(train_labels)
    n_classes = Y_train.shape[1]
    logger.info("Y_train shape: %s  (%d classes)", Y_train.shape, n_classes)

    # ---- TF-IDF -------------------------------------------------------------
    logger.info(
        "Fitting TF-IDF: max_features=%d min_df=%d ngram_range=(1,%d) ...",
        args.max_features, args.min_df, args.ngram_max,
    )
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    logger.info(
        "TF-IDF fit in %.1fs. vocab=%d  X_train=%s  X_test=%s",
        time.time() - t0, len(vec.vocabulary_), X_train.shape, X_test.shape,
    )

    # ---- One-vs-rest classifier ---------------------------------------------
    # OneVsRestClassifier with n_jobs=-1 parallelises across CPU cores. We use
    # LinearSVC by default because for high-dim sparse TF-IDF features it is
    # noticeably faster than LogisticRegression at comparable ranking quality.
    if args.classifier == "linearsvc":
        base = LinearSVC(C=args.C, max_iter=args.max_iter, dual="auto")
    else:
        base = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="liblinear")
    logger.info("Fitting OneVsRestClassifier(%s) on %d classes (n_jobs=%d) ...",
                args.classifier, n_classes, args.n_jobs)
    t0 = time.time()
    clf = OneVsRestClassifier(base, n_jobs=args.n_jobs)
    clf.fit(X_train, Y_train)
    logger.info("Fit complete in %.0fs (%.1f min)", time.time() - t0, (time.time() - t0) / 60)

    # decision_function returns one score per class; rank with no need for a
    # probability calibration step, since we only need a per-record ordering.
    logger.info("Scoring test set ...")
    t0 = time.time()
    scores = clf.decision_function(X_test)
    if scores.ndim == 1:
        # Binary degenerate case (n_classes==1); never hit in practice here.
        scores = scores.reshape(-1, 1)
    logger.info("Scored in %.0fs", time.time() - t0)
    proba = scores  # higher = better; argsort below already uses descending order

    # ---- Predict top-k codes per test record --------------------------------
    logger.info("Picking top-%d codes per test record ...", args.top_k)
    top_idx = np.argsort(-proba, axis=1)[:, : args.top_k]
    code_arr = np.array(mlb.classes_)
    preds = [list(code_arr[row]) for row in top_idx]

    # ---- Build BaselineRecord list ------------------------------------------
    records: list[BaselineRecord] = []
    for i in range(len(test_texts)):
        gold = list(test_gold_codes[i])
        pred = list(preds[i])
        ps, rs, f1 = per_record_prf1(set(pred), set(gold))
        records.append(BaselineRecord(
            parquet_idx=int(test_parquet_idx[i]),
            source_id=test_source_ids[i],
            cod=test_texts[i],
            gold_str=test_gold_str[i],
            gold_codes=gold,
            predicted_codes=pred,
            predicted_codes_invalid=[],
            exact_set_match=set(pred) == set(gold) and bool(pred),
            precision=ps,
            recall=rs,
            f1=f1,
            elapsed_s=0.0,
            cost_usd=0.0,
            error=None,
        ))

    # ---- Save ----------------------------------------------------------------
    write_jsonl(records, results_jsonl)
    save_snapshot(records, snapshot_path)
    m = compute_baseline_metrics(records)
    logger.info("HEADLINE  n=%d  exact=%.4f  macro_f1=%.4f  micro_f1=%.4f  sample_f1=%.4f",
                m["n"], m["exact_set_match_rate"], m["macro_f1"], m["micro_f1"], m["sample_f1"])
    logger.info("Predictions: %s", results_jsonl)
    logger.info("Snapshot:    %s", snapshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
