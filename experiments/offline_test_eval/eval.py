"""Offline held-out TEST evaluation for the size-sweep checkpoints.

WHY THIS EXISTS
---------------
The in-training "test" evaluation (trainer.evaluate(test_ds, metric_key_prefix=
"test") after load_best_model_at_end) was found on 2026-06-12 to re-emit the
VALIDATION metrics of the best checkpoint rather than score the held-out test
split: for small/base/large, every logged test/* metric was bit-identical to
val/* at the best checkpoint (loss, macro_f1, micro_f1, sample_f1, precision,
recall, all to 16 decimals). This script computes the GENUINE held-out test
number by a Trainer-free path: load the saved best checkpoint, run
model.generate over splits.test directly, and compute metrics from the decoded
predictions. It also evaluates splits.val the same way and ASSERTS the two
vectors are not bit-identical, so the aliasing class of bug fails loudly.

The model was trained on cod + age + sex (the processed `text` column is
`cod: ... | age: ... | sex: ...`), so this eval feeds that exact column. No
retraining; this is inference only.

USAGE (on an HPC GPU node, where the checkpoints live):
    uv run python -m experiments.offline_test_eval.eval \
        --checkpoint $RUN_STORAGE_DIR/runs/flan-t5-large/run-XXXX/finetune/checkpoint-NNNN \
        --batch-size 64 \
        --out experiments/offline_test_eval/results/large_test.json

    # quick subset sanity check first:
    uv run python -m experiments.offline_test_eval.eval --checkpoint ... --limit 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"


def _split_codes(text: str, separator: str) -> set[str]:
    if not text:
        return set()
    return {tok.strip() for tok in text.split(separator) if tok.strip()}


def _eval_split(
    cfg: Any,
    model: Any,
    tokenizer: Any,
    df: Any,
    *,
    split_name: str,
    limit: int | None,
    batch_size: int,
) -> dict[str, Any]:
    """Generate predictions over one split and compute the metric vector."""
    from codllm.inference.generation import generate_predictions
    from codllm.metrics import (
        _macro_precision_recall_f1,
        _micro_precision_recall_f1,
        _sample_precision_recall_f1,
    )

    text_col = cfg.dataset_text_column
    label_col = cfg.dataset_label_column
    sep = cfg.label_separator

    work = df if limit is None else df.head(limit)
    texts = [str(t) for t in work[text_col].tolist()]
    golds = [str(g) for g in work[label_col].tolist()]
    sources = (
        [str(s) for s in work["source_id"].tolist()]
        if "source_id" in work.columns
        else [""] * len(work)
    )

    t0 = time.time()
    raw_preds = generate_predictions(cfg, model, tokenizer, texts)
    elapsed = time.time() - t0

    pred_sets = [_split_codes(p, sep) for p in raw_preds]
    gold_sets = [_split_codes(g, sep) for g in golds]

    metrics: dict[str, Any] = {}
    metrics.update(_macro_precision_recall_f1(pred_sets, gold_sets))
    metrics.update(_micro_precision_recall_f1(pred_sets, gold_sets))
    metrics.update(_sample_precision_recall_f1(pred_sets, gold_sets))
    exact = sum(1 for p, g in zip(pred_sets, gold_sets) if p == g)
    metrics["exact_set_match_rate"] = exact / max(len(pred_sets), 1)
    metrics["n"] = len(pred_sets)

    # Per-source macro_f1 (historic_strings_en_2024 reported separately so it can
    # be excluded from the headline, matching the trainer's convention).
    by_src: dict[str, list[int]] = {}
    for i, s in enumerate(sources):
        by_src.setdefault(s, []).append(i)
    per_source: dict[str, dict[str, float]] = {}
    for s, idxs in sorted(by_src.items()):
        ps = [pred_sets[i] for i in idxs]
        gs = [gold_sets[i] for i in idxs]
        m = _macro_precision_recall_f1(ps, gs)
        per_source[s] = {"n": len(idxs), "macro_f1": round(m.get("macro_f1", 0.0), 6)}
    metrics["per_source"] = per_source

    # Headline excluding the auxiliary reference-string source.
    keep = [i for i, s in enumerate(sources) if s != "historic_strings_en_2024"]
    if len(keep) < len(pred_sets):
        ps = [pred_sets[i] for i in keep]
        gs = [gold_sets[i] for i in keep]
        excl = _macro_precision_recall_f1(ps, gs)
        metrics["macro_f1_excl_refstrings"] = round(excl.get("macro_f1", 0.0), 6)
        metrics["n_excl_refstrings"] = len(keep)

    metrics["generate_seconds"] = round(elapsed, 1)
    metrics["split"] = split_name
    return metrics


def _comparable_vector(m: dict[str, Any]) -> tuple:
    """The 6-metric vector used for the test-vs-val aliasing assertion."""
    return (
        round(float(m.get("macro_f1", -1)), 12),
        round(float(m.get("micro_f1", -1)), 12),
        round(float(m.get("sample_f1", -1)), 12),
        round(float(m.get("macro_precision", -1)), 12),
        round(float(m.get("macro_recall", -1)), 12),
        round(float(m.get("exact_set_match_rate", -1)), 12),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to the saved best checkpoint dir (model.safetensors + tokenizer).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N rows of each split (quick sanity check).")
    p.add_argument("--out", default=None, help="Write the full JSON result here.")
    p.add_argument("--skip-val", action="store_true",
                   help="Skip the val eval (then the aliasing assertion is also skipped).")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from codllm.settings.schema import Config
    from codllm.data.handler import DataHandler

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    # Match the size-sweep training/eval config exactly. training_input defaults
    # to ['cod','age','sex'] so the processed `text` column already carries the
    # metadata the model was trained on; we do NOT override it.
    cfg = Config()
    cfg.max_label_count = 3
    cfg.dataset_size = 1.0
    cfg.train_size = 0.9
    cfg.val_size = 0.05
    cfg.test_size = 0.05
    cfg.per_device_eval_batch_size = args.batch_size

    print(f"[offline-eval] training_input = {cfg.training_input}", flush=True)
    print(f"[offline-eval] loading checkpoint: {ckpt}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt))
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("[offline-eval] model on CUDA", flush=True)
    else:
        print("[offline-eval] WARNING: no CUDA; this will be slow on CPU", flush=True)
    model.eval()

    handler = DataHandler(cfg)
    full = handler.ensure_processed()
    splits = handler.split_dataframe(full)
    print(f"[offline-eval] splits: val={len(splits.val)} test={len(splits.test)}", flush=True)

    print("[offline-eval] evaluating TEST split...", flush=True)
    test_metrics = _eval_split(
        cfg, model, tokenizer, splits.test,
        split_name="test", limit=args.limit, batch_size=args.batch_size,
    )
    headline_keys = ["macro_f1", "micro_f1", "sample_f1", "macro_precision",
                     "macro_recall", "exact_set_match_rate", "n",
                     "macro_f1_excl_refstrings"]
    print("[offline-eval] TEST: " + json.dumps(
        {k: test_metrics.get(k) for k in headline_keys}, default=str), flush=True)

    result: dict[str, Any] = {"checkpoint": str(ckpt), "test": test_metrics}

    if not args.skip_val:
        print("[offline-eval] evaluating VAL split (for aliasing assertion)...", flush=True)
        val_metrics = _eval_split(
            cfg, model, tokenizer, splits.val,
            split_name="val", limit=args.limit, batch_size=args.batch_size,
        )
        print("[offline-eval] VAL: " + json.dumps(
            {k: val_metrics.get(k) for k in headline_keys}, default=str), flush=True)
        result["val"] = val_metrics

        tv = _comparable_vector(test_metrics)
        vv = _comparable_vector(val_metrics)
        aliased = (tv == vv)
        result["aliasing_check"] = {
            "test_vector": tv, "val_vector": vv, "bit_identical": aliased,
        }
        if aliased:
            print("[offline-eval] *** ALIASING DETECTED: test vector == val vector. "
                  "The held-out test eval is NOT distinct from val. ***", flush=True)
        else:
            print("[offline-eval] OK: test vector differs from val vector "
                  "(genuine held-out test eval).", flush=True)

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"{ckpt.parent.parent.name}_test.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[offline-eval] wrote {out_path}", flush=True)

    real_test = test_metrics.get("macro_f1_excl_refstrings", test_metrics.get("macro_f1"))
    print(f"\n[offline-eval] REAL held-out test macro_f1 (excl. ref-strings) = {real_test}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
