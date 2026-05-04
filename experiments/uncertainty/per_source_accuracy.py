"""Fast per-source accuracy on the test split — no uncertainty signals.

Strips out the per-step `output_scores` machinery from `inference.py` so we can
batch generation properly and finish in minutes instead of hours. The only
output is per-record (prediction, gold, source_id) and an aggregated
per-source accuracy table.

Usage:
    python -m experiments.uncertainty.per_source_accuracy \\
        --checkpoint experiments/model_from_wandb \\
        --split test --seed 333 \\
        --batch-size 64 \\
        --output experiments/uncertainty/results/per_source_accuracy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("per_source_accuracy")


def _load_eval_split(
    split: str, seed: int,
    *, train_size: float = 0.98, val_size: float = 0.01, test_size: float = 0.01,
) -> list[tuple[str, str, str]]:
    """Same deterministic split as inference.py — returns (text, gold, source_id) triples."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    parquet = Path(__file__).resolve().parents[2] / "data" / "processed" / "data.parquet"
    df = pd.read_parquet(parquet)
    logger.info("Loaded parquet: %d rows", len(df))

    holdout_size = round(val_size + test_size, 10)
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_size, random_state=seed, shuffle=True,
    )
    test_ratio = test_size / holdout_size
    val_df, test_df = train_test_split(
        holdout_df, test_size=test_ratio, random_state=seed, shuffle=True,
    )

    chosen = {"train": train_df, "val": val_df, "test": test_df}[split]
    logger.info(
        "Split %r at seed=%d: train=%d val=%d test=%d (returning %s with %d rows)",
        split, seed, len(train_df), len(val_df), len(test_df), split, len(chosen),
    )
    return [
        (str(row["text"]), str(row["label"]), str(row["source_id"]))
        for _, row in chosen.iterrows()
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=333)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-records", type=int, default=None,
                   help="Cap records (smoke testing). Default: full split.")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (default: cuda if available)")
    p.add_argument("--output", required=True, help="JSON output for per-source accuracy")
    p.add_argument("--predictions-output", default=None,
                   help="Optional JSONL with per-record predictions")
    p.add_argument("--log-every", type=int, default=10, help="Log every N batches")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading model from %s", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    pairs = _load_eval_split(args.split, args.seed)
    if args.max_records is not None:
        pairs = pairs[: args.max_records]
    logger.info("Records: %d, batch_size: %d", len(pairs), args.batch_size)

    pred_handle = None
    if args.predictions_output:
        Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
        pred_handle = open(args.predictions_output, "w", encoding="utf-8")

    per_source_total: dict[str, int] = defaultdict(int)
    per_source_correct: dict[str, int] = defaultdict(int)
    overall_correct = 0
    overall_n = len(pairs)

    t_start = time.time()
    n_batches = (overall_n + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for batch_idx in range(n_batches):
            lo = batch_idx * args.batch_size
            hi = min(lo + args.batch_size, overall_n)
            batch = pairs[lo:hi]
            texts = [t for (t, _, _) in batch]

            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=256,
            ).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for (text, gold, src), pred in zip(batch, preds):
                pred = pred.strip()
                ok = pred == gold.strip()
                per_source_total[src] += 1
                if ok:
                    per_source_correct[src] += 1
                    overall_correct += 1
                if pred_handle is not None:
                    pred_handle.write(json.dumps({
                        "source_id": src,
                        "text": text,
                        "gold": gold,
                        "prediction": pred,
                        "correct": ok,
                    }, ensure_ascii=False) + "\n")

            if (batch_idx + 1) % args.log_every == 0 or batch_idx + 1 == n_batches:
                done = hi
                rate = done / max(time.time() - t_start, 1e-6)
                running_acc = overall_correct / max(done, 1)
                eta_s = (overall_n - done) / max(rate, 1e-6)
                logger.info(
                    "[batch %d/%d] %d/%d  acc=%.4f  rate=%.1f rec/s  eta=%.0fs",
                    batch_idx + 1, n_batches, done, overall_n,
                    running_acc, rate, eta_s,
                )

    if pred_handle is not None:
        pred_handle.close()

    elapsed = time.time() - t_start
    overall_acc = overall_correct / max(overall_n, 1)
    summary = {
        "split": args.split,
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": device,
        "n": overall_n,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_acc,
        "elapsed_s": elapsed,
        "rate_rec_per_s": overall_n / max(elapsed, 1e-6),
        "per_source": {
            src: {
                "n": per_source_total[src],
                "correct": per_source_correct[src],
                "accuracy": per_source_correct[src] / per_source_total[src],
            }
            for src in sorted(per_source_total)
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("Overall accuracy: %d/%d = %.4f", overall_correct, overall_n, overall_acc)
    logger.info("Wallclock: %.1fs (%.1f rec/s)", elapsed, overall_n / max(elapsed, 1e-6))
    logger.info("-" * 70)
    logger.info(f'{"source":<28} {"n":>7} {"correct":>9} {"accuracy":>10}')
    for src, stats in summary["per_source"].items():
        logger.info(f'{src:<28} {stats["n"]:>7} {stats["correct"]:>9} {stats["accuracy"]:>10.4f}')
    logger.info("=" * 70)
    logger.info("Saved to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
