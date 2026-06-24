"""RQ1 baseline: multilingual sentence-embedding similarity with
paraphrase-multilingual-MiniLM-L12-v2.

Differences from the existing CampusAI nomic-embed-text baseline
(experiments.baselines.embedding_baseline):

  * Local model (sentence-transformers), so NO API quota is touched.
  * Designed for cross-lingual semantic transfer: trained on 50+ languages
    including Dutch, French, Danish, Spanish, English. Better suited to
    the multilingual historical corpus than the English-leaning nomic model.
  * Smaller embedding dim (384 vs 768) and faster on CPU.
  * Same evaluation harness: identical test split, same multi-label
    top-k code emission, same metrics.

Use this as the **multilingual semantic transfer** baseline; the
CampusAI nomic baseline is the **English-leaning single-API-call**
baseline. If they give similar numbers, you can drop one in the
thesis; if they differ, that's a finding.

Usage:
    uv run python -m experiments.baselines.multilingual_embed_baseline \\
        --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTERLIST_PATH = REPO_ROOT / "data" / "raw" / "ICD10h_Masterlist_2024.xlsx"
EMBED_CACHE = REPO_ROOT / "data" / "raw" / "paraphrase_minilm_code_embeddings.npz"

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelNAME)s: %(message)s" if False else "%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("multilingual_embed_baseline")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _parse_y_codes(y_codes_field, label_field: str) -> list[str]:
    if y_codes_field is not None:
        try:
            seq = list(y_codes_field)
            if seq:
                return [str(c).strip() for c in seq if str(c).strip()]
        except TypeError:
            pass
    return split_gold(label_field)


def load_masterlist_descriptions() -> tuple[list[str], list[str]]:
    """Return (codes, descriptions) from the ICD-10h masterlist xlsx.

    Targets the canonical columns 'ICD10h' (e.g. 'A00.000') and
    'ICD10hDescription' (e.g. 'Cholera, unspecified') by exact name, with
    a fuzzy fallback if the masterlist is later renamed. The fuzzy match
    used to grab 'IDMasterlist' / 'icd10_2levelCATEGORY' by accident, which
    produces 4-digit integer predictions and category-label descriptions
    -- both of which tank the cosine-similarity baseline to 0% exact match.
    """
    df = pd.read_excel(MASTERLIST_PATH, sheet_name="Masterlist", engine="openpyxl")
    cols = list(df.columns)
    if "ICD10h" in cols and "ICD10hDescription" in cols:
        code_col, desc_col = "ICD10h", "ICD10hDescription"
    else:
        # Fuzzy fallback for a renamed masterlist -- prefer the column that
        # has the *ICD-10h* dot-format codes (e.g. 'A00.000'), not integer
        # row IDs. We detect by sampling the first non-null value.
        code_col = None
        for c in cols:
            sample = df[c].dropna().astype(str).head(5)
            if any("." in s and s[0].isalpha() for s in sample):
                code_col = c
                break
        code_col = code_col or cols[0]
        desc_col = next(
            (c for c in cols if "descr" in c.lower() or c.lower() == "name"),
            cols[1] if len(cols) > 1 else cols[0],
        )
    codes = [str(c).strip() for c in df[code_col] if str(c).strip()]
    descs = [str(d).strip() for d in df[desc_col] if str(d).strip()]
    n = min(len(codes), len(descs))
    return codes[:n], descs[:n]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HuggingFace model id (default {DEFAULT_MODEL})")
    p.add_argument("--label", default="multilingual_minilm_cosine")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda. Sentence-BERT handles MPS too via 'mps'.")
    p.add_argument("--force-rebuild-embeddings", action="store_true")
    args = p.parse_args()

    label = args.label
    results_jsonl = BASELINE_RESULTS_DIR / f"{label}.predictions.jsonl"
    snapshot_path = BASELINE_RESULTS_DIR / f"{label}.results.json"
    test_cache = BASELINE_RESULTS_DIR / f"{label}.test_embeddings.npz"

    logger.info("Loading train + test splits ...")
    train_df, test_df = load_train_and_test()
    logger.info("train=%d  test=%d", len(train_df), len(test_df))

    test_texts = [extract_cod(t) for t in test_df["text"]]
    test_gold_str = list(test_df["label"].astype(str))
    test_gold_codes = [_parse_y_codes(yc, lab) for yc, lab in zip(test_df["y_codes"], test_df["label"])]
    test_source_ids = list(test_df["source_id"].astype(str))
    test_parquet_idx = list(test_df.index.astype(int))

    logger.info("Loading model: %s  (device=%s)", args.model, args.device)
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(args.model, device=args.device)
    logger.info("Model loaded in %.1fs. Embedding dim = %d",
                time.time() - t0, model.get_sentence_embedding_dimension())

    # ---- Code description embeddings (cached) -------------------------------
    codes, descs = load_masterlist_descriptions()
    logger.info("Masterlist: %d codes with descriptions", len(codes))

    code_embeddings = None
    if EMBED_CACHE.exists() and not args.force_rebuild_embeddings:
        z = np.load(EMBED_CACHE)
        if z["embeddings"].shape[0] == len(codes):
            code_embeddings = z["embeddings"]
            logger.info("Loaded cached code embeddings: %s", code_embeddings.shape)

    if code_embeddings is None:
        t0 = time.time()
        code_embeddings = model.encode(
            descs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info("Embedded %d codes in %.1fs", len(descs), time.time() - t0)
        np.savez_compressed(EMBED_CACHE, embeddings=code_embeddings, codes=np.array(codes))
        logger.info("Cached -> %s", EMBED_CACHE)

    # ---- Test set embeddings (cached) --------------------------------------
    test_embeddings = None
    if test_cache.exists() and not args.force_rebuild_embeddings:
        z = np.load(test_cache)
        if z["embeddings"].shape[0] == len(test_texts):
            test_embeddings = z["embeddings"]
            logger.info("Loaded cached test embeddings: %s", test_embeddings.shape)

    if test_embeddings is None:
        t0 = time.time()
        test_embeddings = model.encode(
            test_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info("Embedded %d test records in %.1fs", len(test_texts), time.time() - t0)
        np.savez_compressed(test_cache, embeddings=test_embeddings)

    # ---- Cosine top-k -------------------------------------------------------
    logger.info("Computing top-%d code matches per record ...", args.top_k)
    sim = test_embeddings @ code_embeddings.T
    top_idx = np.argsort(-sim, axis=1)[:, : args.top_k]
    top_scores = np.take_along_axis(sim, top_idx, axis=1)
    code_arr = np.array(codes)
    preds = [list(code_arr[row]) for row in top_idx]

    # ---- Build BaselineRecord list -----------------------------------------
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
            extra={"top_scores": [float(s) for s in top_scores[i]]},
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
