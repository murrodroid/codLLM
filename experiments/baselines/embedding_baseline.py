"""RQ1 baseline: multilingual sentence-embedding similarity.

For each test record's CoD text, embed with the CampusAI nomic-embed-text model
then return the top-k ICD-10h code descriptions whose embedding has the
highest cosine similarity to the input. Multi-label via top-k threshold.

Why this baseline: it is the most realistic 2026-era retrieval baseline. A
historian with an AI assistant that has access to the masterlist would get
behaviour very close to this without any task-specific training. It also
isolates the contribution of FLAN-T5 fine-tuning cleanly: if fine-tuning only
helps a few percent over cosine retrieval, the value of the training data
(and the overall complexity of our approach) deserves scrutiny.

We cache the masterlist embeddings to disk so reruns are cheap.

Usage:
    uv run python -m experiments.baselines.embedding_baseline \\
        --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTERLIST_PATH = REPO_ROOT / "data" / "raw" / "ICD10h_Masterlist_2024.xlsx"
EMBED_CACHE = REPO_ROOT / "data" / "raw" / "nomic_code_embeddings.npz"


def _load_env_file(path: Path) -> None:
    """Minimal `.env` parser. Only used for CAMPUSAI_API_KEY. Real env vars
    take precedence. Keys look like `KEY=value` or `KEY='value'`; comments
    start with `#`; blank lines ignored."""
    if not path.exists():
        return
    key_re = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = key_re.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if (val.startswith("'") and val.endswith("'")) or (
            val.startswith('"') and val.endswith('"')
        ):
            val = val[1:-1]
        os.environ.setdefault(key, val)


# Load .env before any client construction so CAMPUSAI_API_KEY is present
# even when the script is run from a fresh shell. Existing process env wins.
_load_env_file(REPO_ROOT / ".env")

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
logger = logging.getLogger("embedding_baseline")

CAMPUSAI_BASE = "https://api.campusai.compute.dtu.dk/v1"
EMBED_MODEL = "nomic-embed-text"


def _parse_y_codes(y_codes_field, label_field: str) -> list[str]:
    if y_codes_field is not None:
        try:
            seq = list(y_codes_field)
            if seq:
                return [str(c).strip() for c in seq if str(c).strip()]
        except TypeError:
            pass
    return split_gold(label_field)


def _get_campusai_client():
    """Lazy import + construction so the module loads without openai if unused."""
    from openai import OpenAI

    api_key = os.getenv("CAMPUSAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CAMPUSAI_API_KEY not set. Source hpc/env.sh or set the env var."
        )
    return OpenAI(base_url=CAMPUSAI_BASE, api_key=api_key)


def _embed_texts(client, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts using the CampusAI OpenAI-compatible embeddings API.
    Returns an (n, d) float32 array."""
    out: list[list[float]] = []
    n = len(texts)
    log_every = max(1, n // (batch_size * 10))
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        # Strip empty strings -- nomic will reject them.
        batch = [t if t.strip() else " " for t in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for item in resp.data:
            out.append(list(item.embedding))
        if (i // batch_size) % log_every == 0:
            logger.info("  embedded %d / %d  elapsed=%.1fs", min(i + batch_size, n), n, time.time() - t0)
    return np.asarray(out, dtype=np.float32)


def load_masterlist_descriptions() -> tuple[list[str], list[str]]:
    """Return (codes, descriptions) from the ICD-10h masterlist xlsx.

    Targets 'ICD10h' + 'ICD10hDescription' by exact name; fuzzy fallback
    avoids the prior bug where 'IDMasterlist' / 'icd10_2levelCATEGORY' were
    selected instead and the baseline produced 0% exact match.
    """
    df = pd.read_excel(MASTERLIST_PATH, sheet_name="Masterlist", engine="openpyxl")
    cols = list(df.columns)
    if "ICD10h" in cols and "ICD10hDescription" in cols:
        code_col, desc_col = "ICD10h", "ICD10hDescription"
    else:
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
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of top-similarity code descriptions to emit per record.")
    p.add_argument("--label", default="embedding_cosine_nomic")
    p.add_argument("--force-rebuild-embeddings", action="store_true")
    args = p.parse_args()

    label = args.label
    results_jsonl = BASELINE_RESULTS_DIR / f"{label}.predictions.jsonl"
    snapshot_path = BASELINE_RESULTS_DIR / f"{label}.results.json"

    logger.info("Loading train + test splits ...")
    train_df, test_df = load_train_and_test()
    logger.info("train=%d  test=%d", len(train_df), len(test_df))

    test_texts = [extract_cod(t) for t in test_df["text"]]
    test_gold_str = list(test_df["label"].astype(str))
    test_gold_codes = [_parse_y_codes(yc, lab) for yc, lab in zip(test_df["y_codes"], test_df["label"])]
    test_source_ids = list(test_df["source_id"].astype(str))
    test_parquet_idx = list(test_df.index.astype(int))

    # ---- Masterlist code description embeddings (cached) -------------------
    codes, descs = load_masterlist_descriptions()
    logger.info("Masterlist: %d codes with descriptions", len(codes))

    code_embeddings = None
    if EMBED_CACHE.exists() and not args.force_rebuild_embeddings:
        z = np.load(EMBED_CACHE)
        if z["embeddings"].shape[0] == len(codes):
            code_embeddings = z["embeddings"]
            logger.info("Loaded cached code embeddings: %s", code_embeddings.shape)

    if code_embeddings is None:
        client = _get_campusai_client()
        logger.info("Embedding %d masterlist code descriptions with %s ...",
                    len(descs), EMBED_MODEL)
        code_embeddings = _embed_texts(client, descs)
        # L2-normalise so cosine similarity == dot product.
        norms = np.linalg.norm(code_embeddings, axis=1, keepdims=True) + 1e-9
        code_embeddings = code_embeddings / norms
        np.savez_compressed(EMBED_CACHE, embeddings=code_embeddings, codes=np.array(codes))
        logger.info("Cached embeddings -> %s", EMBED_CACHE)

    # ---- Test-set embeddings (in batches, with checkpointing) --------------
    # Cached per shard of 5000 to survive quota walls.
    test_cache = BASELINE_RESULTS_DIR / f"{label}.test_embeddings.npz"
    if test_cache.exists():
        z = np.load(test_cache)
        if z["embeddings"].shape[0] == len(test_texts):
            test_embeddings = z["embeddings"]
            logger.info("Loaded cached test embeddings: %s", test_embeddings.shape)
        else:
            test_embeddings = None
    else:
        test_embeddings = None
    if test_embeddings is None:
        client = _get_campusai_client()
        logger.info("Embedding %d test records ...", len(test_texts))
        test_embeddings = _embed_texts(client, test_texts, batch_size=128)
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-9
        test_embeddings = test_embeddings / norms
        np.savez_compressed(test_cache, embeddings=test_embeddings)
        logger.info("Cached test embeddings -> %s", test_cache)

    # ---- Cosine-similarity top-k -------------------------------------------
    logger.info("Computing top-%d code matches per record ...", args.top_k)
    # sim shape: (n_test, n_codes)
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
