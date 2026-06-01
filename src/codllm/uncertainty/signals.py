"""Per-record uncertainty signals from greedy seq2seq generation.

Mirrors the contract of ``experiments/uncertainty/inference.py`` so the offline
analysis notebooks and the production end-of-training callback agree on what
``mean_logprob`` etc. mean. Greedy decoding keeps the per-step argmax aligned
with the generated sequence for clean bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class UncertaintySignals:
    """Per-record uncertainty bundle from one greedy generation."""

    prediction: str
    n_tokens: int
    sum_logprob: float
    mean_logprob: float
    min_logprob: float
    mean_entropy: float
    first_token_entropy: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the signals."""
        return {
            "prediction": self.prediction,
            "n_tokens": self.n_tokens,
            "sum_logprob": self.sum_logprob,
            "mean_logprob": self.mean_logprob,
            "min_logprob": self.min_logprob,
            "mean_entropy": self.mean_entropy,
            "first_token_entropy": self.first_token_entropy,
        }


# Sentinel values for generations that produce zero usable tokens (immediate EOS,
# pad-only output, etc.). 0.0 across the board would have ranked these AS THE
# MOST CONFIDENT outputs on RC curves (since 0 > any typical -2 to -3 logprob),
# poisoning the high-coverage buckets. -1e9 / +1e9 keep them at the minimum
# confidence end after orientation while remaining JSON-safe (no inf/NaN).
_EMPTY_LOGPROB_SENTINEL = -1e9
_EMPTY_ENTROPY_SENTINEL = 1e9
_EMPTY_SIGNALS = UncertaintySignals(
    prediction="",
    n_tokens=0,
    sum_logprob=_EMPTY_LOGPROB_SENTINEL,
    mean_logprob=_EMPTY_LOGPROB_SENTINEL,
    min_logprob=_EMPTY_LOGPROB_SENTINEL,
    mean_entropy=_EMPTY_ENTROPY_SENTINEL,
    first_token_entropy=_EMPTY_ENTROPY_SENTINEL,
)


def _resolve_decoder_start_token_id(model: Any, tokenizer: Any) -> int | None:
    """Return the decoder start token id used by encoder-decoder generation."""
    decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_id is not None:
        return int(decoder_start_id)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    return None


def _strip_decoder_start(sequence: torch.Tensor, decoder_start_id: int | None) -> torch.Tensor:
    """Drop the leading decoder_start_token_id when present."""
    if sequence.numel() == 0:
        return sequence
    if decoder_start_id is None:
        return sequence
    if int(sequence[0].item()) == decoder_start_id:
        return sequence[1:]
    return sequence


def _step_entropy(log_probs: torch.Tensor) -> float:
    """Numerical-stable Shannon entropy of one softmax distribution."""
    probs = log_probs.exp()
    entropy = -(probs * log_probs).nan_to_num(nan=0.0).sum().item()
    return float(entropy)


def _aggregate_signals(
    decoded: str,
    token_logprobs: list[float],
    token_entropies: list[float],
) -> UncertaintySignals:
    """Build an UncertaintySignals dataclass from per-token streams."""
    if not token_logprobs:
        return UncertaintySignals(
            prediction=decoded,
            n_tokens=0,
            sum_logprob=_EMPTY_LOGPROB_SENTINEL,
            mean_logprob=_EMPTY_LOGPROB_SENTINEL,
            min_logprob=_EMPTY_LOGPROB_SENTINEL,
            mean_entropy=_EMPTY_ENTROPY_SENTINEL,
            first_token_entropy=_EMPTY_ENTROPY_SENTINEL,
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


def compute_per_record_signals(
    model: Any,
    tokenizer: Any,
    source_text: str,
    *,
    device: str,
    max_new_tokens: int,
) -> UncertaintySignals:
    """Run greedy generation on one source string and return uncertainty signals.

    ``output_scores=True, return_dict_in_generate=True`` is required for
    per-step logits; greedy keeps the argmax decoder aligned with the per-step
    distribution so the bookkeeping stays simple. Returns zeroed signals when
    generation produces no usable tokens (e.g. an immediate EOS or pad).
    """
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    decoder_start_id = _resolve_decoder_start_token_id(model, tokenizer)
    sequence = output.sequences[0]
    generated_ids = _strip_decoder_start(sequence, decoder_start_id)

    if generated_ids.numel() == 0:
        return _EMPTY_SIGNALS

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    token_logprobs: list[float] = []
    token_entropies: list[float] = []

    for step_idx, step_scores in enumerate(output.scores):
        if step_idx >= generated_ids.shape[0]:
            break
        token_id = int(generated_ids[step_idx].item())
        if token_id == eos_id or token_id == pad_id:
            break

        log_probs = torch.log_softmax(step_scores[0], dim=-1)
        token_logprobs.append(float(log_probs[token_id].item()))
        token_entropies.append(_step_entropy(log_probs))

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _aggregate_signals(decoded, token_logprobs, token_entropies)
