"""Per-record uncertainty inference for trained codllm seq2seq checkpoints.

Loads a fine-tuned Flan-T5 checkpoint, runs greedy generation with output_scores
enabled over a labeled eval split, and writes per-record JSONL containing the
prediction, the gold label, and a battery of uncertainty signals:

  * sum_logprob          sum of token log-probs of the generated sequence
  * mean_logprob         length-normalized sum (the standard sequence score)
  * min_logprob          weakest token in the sequence (catches local hesitation)
  * mean_entropy         mean Shannon entropy of the model's output distribution
                         at each generation step
  * first_token_entropy  entropy at step 1 only (cheap, often correlates with the
                         whole sequence — useful sanity check)

These are the inputs to:
  * post-hoc temperature scaling for calibration (Guo et al. 2017)
  * risk-coverage curves / AURC for selective prediction
  * threshold-tuned "human-in-the-loop required here" detection

The output JSONL is the only artifact this script produces — all calibration
and selective-prediction analysis runs on it as plain pandas/sklearn code.

Usage:
    python -m experiments.uncertainty.inference \\
        --checkpoint /dtu/blackhole/.../runs/run-XXXX/checkpoint-YYYY \\
        --split val \\
        --output experiments/uncertainty/results/val_run-XXXX.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("uncertainty.inference")


# ---------------------------------------------------------------------------
# Per-record uncertainty computation
# ---------------------------------------------------------------------------


@dataclass
class UncertaintySignals:
    prediction: str
    n_tokens: int
    sum_logprob: float
    mean_logprob: float
    min_logprob: float
    mean_entropy: float
    first_token_entropy: float


def _per_record_signals(
    model,
    tokenizer,
    source_text: str,
    *,
    device: str,
    max_new_tokens: int,
) -> UncertaintySignals:
    """Run greedy generation on one source string and collect token-level signals.

    Greedy is used so that the generated sequence and the per-step argmax
    distributions are aligned. With sampling/beam search the bookkeeping for
    "which token's logprob am I summing" gets messier without changing the
    headline number.
    """
    inputs = tokenizer(
        source_text, return_tensors="pt", truncation=True
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    sequence = output.sequences[0]
    # For encoder-decoder models, generate() returns ONLY the generated tokens
    # in `sequences` (no input prefix), so generated_ids = sequences[0] minus
    # the leading decoder_start_token_id.
    decoder_start_id = (
        model.config.decoder_start_token_id
        if model.config.decoder_start_token_id is not None
        else tokenizer.pad_token_id
    )
    if sequence[0].item() == decoder_start_id:
        generated_ids = sequence[1:]
    else:
        generated_ids = sequence

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    token_logprobs: list[float] = []
    token_entropies: list[float] = []

    for step_idx, step_scores in enumerate(output.scores):
        # step_scores: (batch=1, vocab_size). These are raw logits for that step.
        if step_idx >= generated_ids.shape[0]:
            break
        token_id = generated_ids[step_idx].item()
        # Stop accumulating once we hit EOS or padding (sequence is over).
        if token_id == eos_id or token_id == pad_id:
            break

        logits = step_scores[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Numerical-stability-aware entropy: -sum(p * log p), zero where p==0.
        entropy = float(-(probs * log_probs).nan_to_num(nan=0.0).sum().item())
        chosen_logprob = float(log_probs[token_id].item())

        token_logprobs.append(chosen_logprob)
        token_entropies.append(entropy)

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not token_logprobs:
        return UncertaintySignals(
            prediction=decoded,
            n_tokens=0,
            sum_logprob=0.0,
            mean_logprob=0.0,
            min_logprob=0.0,
            mean_entropy=0.0,
            first_token_entropy=0.0,
        )

    return UncertaintySignals(
        prediction=decoded,
        n_tokens=len(token_logprobs),
        sum_logprob=float(sum(token_logprobs)),
        mean_logprob=float(np.mean(token_logprobs)),
        min_logprob=float(min(token_logprobs)),
        mean_entropy=float(np.mean(token_entropies)),
        first_token_entropy=float(token_entropies[0]),
    )


# ---------------------------------------------------------------------------
# Eval-split loading (delegates to the codllm data layer for consistency)
# ---------------------------------------------------------------------------


def _load_eval_split(
    split: str,
    max_records: int | None,
    seed: int,
    *,
    train_size: float = 0.98,
    val_size: float = 0.01,
    test_size: float = 0.01,
    parquet_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Return [(source_text, gold_label, source_id), ...] from a deterministic split.

    Reads `data/processed/data.parquet` directly and replicates
    `DataHandler.split_dataframe` (sklearn `train_test_split` with the configured
    seed and split sizes), bypassing `ensure_processed` so this script is
    branch-agnostic — no metadata-version mismatch can trigger a parquet rebuild
    with shifted row identities.

    The on-disk parquet was produced with `max_label_count=1`, so every row
    already has exactly one code; no post-split single-CoD filter is needed.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    parquet = parquet_path or (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "data.parquet"
    )
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found at {parquet}")

    df = pd.read_parquet(parquet)
    logger.info("Loaded parquet: %d rows from %s", len(df), parquet)

    holdout_size = round(val_size + test_size, 10)
    if holdout_size <= 0:
        train_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_df = df.iloc[0:0].copy()
        test_df = df.iloc[0:0].copy()
    else:
        train_df, holdout_df = train_test_split(
            df, test_size=holdout_size, random_state=seed, shuffle=True,
        )
        if val_size == 0:
            val_df = df.iloc[0:0].copy()
            test_df = holdout_df
        elif test_size == 0:
            val_df = holdout_df
            test_df = df.iloc[0:0].copy()
        else:
            test_ratio = test_size / holdout_size
            val_df, test_df = train_test_split(
                holdout_df, test_size=test_ratio, random_state=seed, shuffle=True,
            )

    if split == "val":
        chosen = val_df
    elif split == "test":
        chosen = test_df
    elif split == "train":
        chosen = train_df
    else:
        raise ValueError(f"Unknown split: {split!r}")

    logger.info(
        "Split %r at seed=%d: train=%d val=%d test=%d (returning %s with %d rows)",
        split, seed, len(train_df), len(val_df), len(test_df), split, len(chosen),
    )

    pairs = [
        (str(row["text"]), str(row["label"]), str(row["source_id"]))
        for _, row in chosen.iterrows()
    ]
    if max_records is not None:
        pairs = pairs[:max_records]
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint dir")
    p.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Which split to score (val by default, since we calibrate on val)"
    )
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--max-records", type=int, default=None,
        help="Cap records (useful for smoke tests; None = full split)"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=32,
        help="Generation length cap (matches CODLLM_MAX_TARGET_LENGTH default)"
    )
    p.add_argument(
        "--device", default=None,
        help="cuda / cuda:0 / cpu (defaults to cuda if available)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="data_seed used to reconstruct the split. MUST match the value the "
             "model was trained with (e.g. 333 for the perturbation-overhaul runs)."
    )
    p.add_argument("--train-size", type=float, default=0.98)
    p.add_argument("--val-size", type=float, default=0.01)
    p.add_argument("--test-size", type=float, default=0.01)
    p.add_argument(
        "--log-every", type=int, default=50,
        help="Log progress every N records"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer + model from %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path)).to(device)
    model.eval()

    logger.info(
        "Loading %s split (seed=%d, sizes=%.3f/%.3f/%.3f)...",
        args.split, args.seed, args.train_size, args.val_size, args.test_size,
    )
    pairs = _load_eval_split(
        args.split, args.max_records, args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    logger.info("Loaded %d records.", len(pairs))

    t_start = time.time()
    n_correct = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (source_text, gold, source_id) in enumerate(pairs):
            sig = _per_record_signals(
                model,
                tokenizer,
                source_text,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            correct = sig.prediction.strip() == gold.strip()
            if correct:
                n_correct += 1
            row = {
                "idx": idx,
                "source": source_text,
                "source_id": source_id,
                "gold": gold,
                "prediction": sig.prediction,
                "correct": correct,
                "n_tokens": sig.n_tokens,
                "sum_logprob": sig.sum_logprob,
                "mean_logprob": sig.mean_logprob,
                "min_logprob": sig.min_logprob,
                "mean_entropy": sig.mean_entropy,
                "first_token_entropy": sig.first_token_entropy,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (idx + 1) % args.log_every == 0:
                rate = (idx + 1) / (time.time() - t_start)
                logger.info(
                    "[%d/%d] running_acc=%.3f  rate=%.1f rec/s",
                    idx + 1, len(pairs), n_correct / (idx + 1), rate,
                )

    elapsed = time.time() - t_start
    logger.info(
        "Done. n=%d  acc=%.4f  elapsed=%.1fs  rate=%.1f rec/s",
        len(pairs), n_correct / max(len(pairs), 1), elapsed, len(pairs) / max(elapsed, 1),
    )
    logger.info("Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
