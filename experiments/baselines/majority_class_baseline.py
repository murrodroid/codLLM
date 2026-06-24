"""RQ1 baseline: predict the top-k most-common training-set codes for every test record.

The simplest possible "no learning" floor. Uses the codes-by-frequency
distribution from the training split to predict top-k codes for every test
record, regardless of input. This baseline quantifies how predictable the
label distribution alone is -- a useful sanity check that any non-trivial
approach beats.

Also produces a **per-source** breakdown, because if the most common
training code is e.g. "R56.800" (unspecified convulsions) but the
Copenhagen 1850s corpus is dominated by infectious disease, this
baseline's per-source macro_f1 will be uniformly bad -- informative.

Run:
    uv run python -m experiments.baselines.majority_class_baseline
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np

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
logger = logging.getLogger("majority_class_baseline")


def _parse_y_codes(y_codes_field, label_field: str) -> list[str]:
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
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--label", default="majority_class_topk")
    args = p.parse_args()

    label = args.label
    results_jsonl = BASELINE_RESULTS_DIR / f"{label}.predictions.jsonl"
    snapshot_path = BASELINE_RESULTS_DIR / f"{label}.results.json"

    logger.info("Loading train + test splits ...")
    train_df, test_df = load_train_and_test()
    logger.info("train=%d  test=%d", len(train_df), len(test_df))

    # ---- Count training-set code frequencies --------------------------------
    code_counter: Counter = Counter()
    for yc, lab in zip(train_df["y_codes"], train_df["label"]):
        codes = _parse_y_codes(yc, lab)
        for c in codes:
            code_counter[c] += 1
    top_codes = [c for c, _ in code_counter.most_common(args.top_k)]
    top_with_counts = list(code_counter.most_common(args.top_k))
    logger.info("Top-%d most common training codes: %s",
                args.top_k, ", ".join(f"{c}({n})" for c, n in top_with_counts))

    # ---- Apply to test split -----------------------------------------------
    test_texts = [extract_cod(t) for t in test_df["text"]]
    test_gold_str = list(test_df["label"].astype(str))
    test_gold_codes = [_parse_y_codes(yc, lab) for yc, lab in zip(test_df["y_codes"], test_df["label"])]
    test_source_ids = list(test_df["source_id"].astype(str))
    test_parquet_idx = list(test_df.index.astype(int))

    records: list[BaselineRecord] = []
    for i in range(len(test_texts)):
        gold = list(test_gold_codes[i])
        # Predict the same top-k for every record, regardless of input.
        pred = list(top_codes)
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
            extra={"top_k_codes": list(top_with_counts)},
        ))

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
