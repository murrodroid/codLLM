"""Standalone local GPU inference over a dumped texts file (BATCHED).

Self-contained: needs only torch + transformers + numpy (no codllm, no dataset).
Reads the test-split texts produced by dump_test_split.py and writes the same
per-record JSONL schema as inference.py (gold/gold_codes/prediction/pred_codes/
correct + the five uncertainty signals).

Batched generation keeps the GPU busy (batch-1 sat at ~5% utilization). The
per-record uncertainty signals are recovered from the batched output_scores with
the same math as the batch-1 path, vectorized over the batch. A --verify mode
checks the batched signals against the batch-1 reference on a sample.

Run from the repo root with the scratch GPU venv:
    .venv-gpu/Scripts/python.exe -m experiments.uncertainty.local_infer \
        --checkpoint <local ckpt> --texts <texts.jsonl> --output <out.jsonl> \
        --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class UncertaintySignals:
    prediction: str
    n_tokens: int
    sum_logprob: float
    mean_logprob: float
    min_logprob: float
    mean_entropy: float
    first_token_entropy: float


def _split_codes(text: str, separator: str) -> set[str]:
    return {tok.strip() for tok in str(text).split(separator) if tok.strip()}


def _decoder_start(model, tokenizer) -> int:
    d = model.config.decoder_start_token_id
    return d if d is not None else tokenizer.pad_token_id


def _per_record_signals(model, tokenizer, source_text, *, device, max_new_tokens):
    """Batch-1 reference (used by --verify); identical math to the HPC inference.py."""
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0]
    ds = _decoder_start(model, tokenizer)
    gen = seq[1:] if seq[0].item() == ds else seq
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    lps, ents = [], []
    for t, step in enumerate(out.scores):
        if t >= gen.shape[0]:
            break
        tid = gen[t].item()
        if tid == eos_id or tid == pad_id:
            break
        lp = torch.log_softmax(step[0], dim=-1)
        ent = float(-(lp.exp() * lp).nan_to_num(nan=0.0).sum().item())
        lps.append(float(lp[tid].item()))
        ents.append(ent)
    decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()
    if not lps:
        return UncertaintySignals(decoded, 0, 0., 0., 0., 0., 0.)
    return UncertaintySignals(decoded, len(lps), float(sum(lps)), float(np.mean(lps)),
                              float(min(lps)), float(np.mean(ents)), float(ents[0]))


def _batch_signals(model, tokenizer, texts, *, device, max_new_tokens):
    """Vectorized batched signals. Returns one UncertaintySignals per input text."""
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True,
                       truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, output_scores=True, return_dict_in_generate=True)
    seqs = out.sequences  # (B, L)
    B = seqs.shape[0]
    ds = _decoder_start(model, tokenizer)
    gen = seqs[:, 1:] if bool((seqs[:, 0] == ds).all()) else seqs  # (B, Sgen)
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id

    sum_lp = torch.zeros(B, device=device)
    sum_ent = torch.zeros(B, device=device)
    min_lp = torch.full((B,), float("inf"), device=device)
    first_ent = torch.zeros(B, device=device)
    n_tok = torch.zeros(B, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(len(out.scores)):
        if t >= gen.shape[1]:
            break
        tok_t = gen[:, t]                                   # (B,)
        is_stop = (tok_t == eos_id) | (tok_t == pad_id)
        active = (~finished) & (~is_stop)
        af = active.float()
        lp = torch.log_softmax(out.scores[t], dim=-1)        # (B, V)
        chosen = lp.gather(1, tok_t.unsqueeze(1)).squeeze(1)  # (B,)
        ent = -(lp.exp() * lp).nan_to_num(nan=0.0).sum(dim=-1)  # (B,)
        sum_lp = sum_lp + chosen * af
        sum_ent = sum_ent + ent * af
        min_lp = torch.where(active, torch.minimum(min_lp, chosen), min_lp)
        first_ent = torch.where(active & (n_tok == 0), ent, first_ent)
        n_tok = n_tok + af
        finished = finished | is_stop

    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    has = n_tok > 0
    mean_lp = torch.where(has, sum_lp / n_tok.clamp_min(1), torch.zeros_like(sum_lp))
    mean_ent = torch.where(has, sum_ent / n_tok.clamp_min(1), torch.zeros_like(sum_ent))
    min_lp = torch.where(has, min_lp, torch.zeros_like(min_lp))
    nt = n_tok.int().tolist()
    sl, ml, mnl, me, fe = (sum_lp.tolist(), mean_lp.tolist(), min_lp.tolist(),
                           mean_ent.tolist(), first_ent.tolist())
    out_list = []
    for b in range(B):
        d = decoded[b].strip()
        if nt[b] == 0:
            out_list.append(UncertaintySignals(d, 0, 0., 0., 0., 0., 0.))
        else:
            out_list.append(UncertaintySignals(d, nt[b], sl[b], ml[b], mnl[b], me[b], fe[b]))
    return out_list


def _load(checkpoint, device):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.eval()
    return tok, model


def _verify(model, tok, texts, *, device, max_new_tokens):
    batched = _batch_signals(model, tok, texts, device=device, max_new_tokens=max_new_tokens)
    worst_pred, worst_mlp = 0, 0.0
    pred_match = 0
    for txt, bs in zip(texts, batched):
        ref = _per_record_signals(model, tok, txt, device=device, max_new_tokens=max_new_tokens)
        pred_match += int(bs.prediction == ref.prediction)
        worst_mlp = max(worst_mlp, abs(bs.mean_logprob - ref.mean_logprob))
    print(f"VERIFY n={len(texts)}: predictions identical {pred_match}/{len(texts)}; "
          f"max |mean_logprob diff| = {worst_mlp:.2e}", flush=True)
    return pred_match == len(texts) and worst_mlp < 1e-2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--texts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--label-separator", default=",")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--verify", type=int, default=0,
                   help="If >0, compare batched vs batch-1 on the first N records and exit.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available; refusing to run a full split on CPU.", flush=True)
        return 3

    rows = [json.loads(line) for line in Path(args.texts).open(encoding="utf-8")]
    if args.max_records is not None:
        rows = rows[:args.max_records]

    print(f"Loading {args.checkpoint} (capability {torch.cuda.get_device_capability(0)})", flush=True)
    tok, model = _load(args.checkpoint, device)

    if args.verify:
        texts = [str(r["text"]) for r in rows[:args.verify]]
        ok = _verify(model, tok, texts, device=device, max_new_tokens=args.max_new_tokens)
        print("VERIFY:", "PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    sep = args.label_separator
    t0 = time.time()
    with outp.open("w", encoding="utf-8") as f:  # always overwrite; fresh, exact-count output
        idx = 0
        while idx < len(rows):
            batch = rows[idx:idx + args.batch_size]
            sigs = _batch_signals(model, tok, [str(r["text"]) for r in batch],
                                  device=device, max_new_tokens=args.max_new_tokens)
            for r, sig in zip(batch, sigs):
                pc = _split_codes(sig.prediction, sep)
                gc = _split_codes(str(r["gold"]), sep)
                f.write(json.dumps({
                    "idx": idx, "source": r["text"], "source_id": r.get("source_id", ""),
                    "gold": r["gold"], "gold_codes": sorted(gc),
                    "prediction": sig.prediction, "pred_codes": sorted(pc),
                    "correct": pc == gc, "n_tokens": sig.n_tokens,
                    "sum_logprob": sig.sum_logprob, "mean_logprob": sig.mean_logprob,
                    "min_logprob": sig.min_logprob, "mean_entropy": sig.mean_entropy,
                    "first_token_entropy": sig.first_token_entropy,
                }, ensure_ascii=False) + "\n")
                idx += 1
            f.flush()
            if idx % args.log_every < args.batch_size:
                rate = idx / (time.time() - t0)
                print(f"[{idx}/{len(rows)}] rate={rate:.0f} rec/s", flush=True)

    print(f"Done. {len(rows)} rows, {time.time() - t0:.0f}s, "
          f"{len(rows) / max(time.time() - t0, 1):.0f} rec/s -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
