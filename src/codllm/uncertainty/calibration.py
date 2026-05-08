"""Post-hoc temperature scaling (Guo et al. 2017) for seq2seq next-token logits.

A single learned scalar `T > 0` rescales raw logits before softmax to minimize
validation NLL. Argmax is preserved, so accuracy and exact_match are unchanged
and quantile-based RC curves are invariant. We still fit and persist T because:

  * "We applied temperature scaling on val" is a one-line methodology bullet
    that lets reviewers stop poking at calibration.
  * Absolute-threshold use cases (e.g. "hand to human if p<0.9") need
    calibrated probabilities, even if the quantile RC curve does not.

Fit uses teacher-forced forward passes (one per record), which is much
cheaper than autoregressive generation, and LBFGS for the 1-D optimization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class TemperatureFit:
    """Result of fitting a temperature scalar on a validation set."""

    temperature: float
    nll_before: float
    nll_after: float
    n_tokens: int
    n_records: int
    max_records: int | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the fit summary."""
        return {
            "temperature": float(self.temperature),
            "nll_before": float(self.nll_before),
            "nll_after": float(self.nll_after),
            "n_tokens": int(self.n_tokens),
            "n_records": int(self.n_records),
            "max_records": (int(self.max_records) if self.max_records else None),
        }


def fit_temperature(
    logits: torch.Tensor,
    gold_ids: torch.Tensor,
    *,
    max_iter: int = 100,
    initial_temperature: float = 1.0,
) -> tuple[float, float, float]:
    """Fit T>0 minimizing cross-entropy of (logits/T) against gold_ids.

    Returns (temperature, nll_before, nll_after). Optimizes log T so the
    constraint T>0 is enforced implicitly.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D tensor of shape [N, vocab].")
    if gold_ids.ndim != 1 or gold_ids.shape[0] != logits.shape[0]:
        raise ValueError("gold_ids must be a 1-D tensor matching logits batch size.")
    if logits.shape[0] == 0:
        return float(initial_temperature), 0.0, 0.0

    device = logits.device
    cross_entropy = torch.nn.functional.cross_entropy

    nll_before = float(cross_entropy(logits, gold_ids).item())

    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    log_temperature.data.fill_(float(torch.log(torch.tensor(initial_temperature)).item()))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = log_temperature.exp()
        scaled_logits = logits / scale
        loss = cross_entropy(scaled_logits, gold_ids)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        scale = log_temperature.exp()
        nll_after = float(
            cross_entropy(logits / scale, gold_ids).item()
        )
        return float(scale.item()), nll_before, nll_after


# ---------------------------------------------------------------------------
# Validation-set logits collection (teacher-forced forward passes)
# ---------------------------------------------------------------------------


def _collect_logits_for_record(
    model: Any,
    tokenizer: Any,
    source_text: str,
    target_text: str,
    *,
    device: str,
    max_source_length: int,
    max_target_length: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Forward one (source, target) pair and return per-token (logits, gold_ids).

    Returns None when the target tokenizes to zero non-padding tokens.
    """
    inputs = tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    ).to(device)
    targets = tokenizer(
        text_target=target_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_target_length,
    ).to(device)
    label_ids = targets["input_ids"][0]
    if label_ids.numel() == 0:
        return None

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=label_ids.unsqueeze(0),
        )
    logits = outputs.logits[0]
    if logits.shape[0] != label_ids.shape[0]:
        # Defensive: trim to the shorter of the two so they align.
        cutoff = min(logits.shape[0], label_ids.shape[0])
        logits = logits[:cutoff]
        label_ids = label_ids[:cutoff]

    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        keep_mask = label_ids != pad_id
        logits = logits[keep_mask]
        label_ids = label_ids[keep_mask]

    if label_ids.numel() == 0:
        return None
    return logits.detach(), label_ids.detach()


def collect_val_logits(
    model: Any,
    tokenizer: Any,
    val_df: pd.DataFrame,
    *,
    device: str,
    max_source_length: int,
    max_target_length: int,
    text_column: str,
    label_column: str,
    max_records: int | None = 2000,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Collect per-token (logits, gold_ids) tensors for up to `max_records` rows."""
    if val_df is None or len(val_df) == 0:
        return (
            torch.zeros((0,), device=device),
            torch.zeros((0,), device=device, dtype=torch.long),
            0,
        )

    selected = val_df
    if max_records is not None and len(val_df) > max_records:
        selected = val_df.head(max_records)

    logit_chunks: list[torch.Tensor] = []
    gold_chunks: list[torch.Tensor] = []
    n_records_used = 0
    for _, row in selected.iterrows():
        if pd.isna(row.get(text_column)) or pd.isna(row.get(label_column)):
            continue
        source_text = str(row[text_column])
        gold = str(row[label_column])
        result = _collect_logits_for_record(
            model,
            tokenizer,
            source_text,
            gold,
            device=device,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        if result is None:
            continue
        logits, gold_ids = result
        logit_chunks.append(logits)
        gold_chunks.append(gold_ids)
        n_records_used += 1

    if not logit_chunks:
        return (
            torch.zeros((0,), device=device),
            torch.zeros((0,), device=device, dtype=torch.long),
            0,
        )

    return (
        torch.cat(logit_chunks, dim=0),
        torch.cat(gold_chunks, dim=0).to(torch.long),
        n_records_used,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def fit_temperature_on_val(
    model: Any,
    tokenizer: Any,
    val_df: pd.DataFrame,
    *,
    device: str,
    max_source_length: int,
    max_target_length: int,
    text_column: str,
    label_column: str,
    max_records: int | None = 2000,
) -> TemperatureFit | None:
    """Run the full val-pass + LBFGS fit; returns None when val is empty."""
    if val_df is None or len(val_df) == 0:
        return None
    if not hasattr(model, "forward"):
        return None

    was_training = getattr(model, "training", False)
    if was_training:
        model.eval()
    try:
        logits, gold_ids, n_records = collect_val_logits(
            model,
            tokenizer,
            val_df,
            device=device,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            text_column=text_column,
            label_column=label_column,
            max_records=max_records,
        )
    finally:
        if was_training:
            model.train()

    if logits.shape[0] == 0:
        return None

    temperature, nll_before, nll_after = fit_temperature(logits, gold_ids)
    return TemperatureFit(
        temperature=temperature,
        nll_before=nll_before,
        nll_after=nll_after,
        n_tokens=int(logits.shape[0]),
        n_records=n_records,
        max_records=max_records,
    )


def write_temperature_json(fit: TemperatureFit, path: Path) -> Path:
    """Persist a temperature fit summary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fit.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def log_temperature_to_wandb(fit: TemperatureFit, *, namespace: str = "uncertainty") -> None:
    """Log temperature scalar + before/after NLL to the current W&B run."""
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return
    wandb.log(
        {
            f"{namespace}/temperature": float(fit.temperature),
            f"{namespace}/temperature_nll_before": float(fit.nll_before),
            f"{namespace}/temperature_nll_after": float(fit.nll_after),
            f"{namespace}/temperature_n_tokens": int(fit.n_tokens),
        }
    )
