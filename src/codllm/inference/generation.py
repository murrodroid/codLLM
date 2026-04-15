from __future__ import annotations

from typing import Any, Mapping

import torch

from codllm.config import Config


def _batched_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """Split texts into stable batches for model inference."""
    if batch_size < 1:
        raise ValueError("Inference batch_size must be at least 1.")
    return [texts[idx : idx + batch_size] for idx in range(0, len(texts), batch_size)]


def _resolve_model_device(model: Any, fallback: torch.device) -> torch.device:
    """Resolve the torch device that should receive inference tensors."""
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_parameter = next(parameters())
        except (StopIteration, TypeError):
            return fallback
        return first_parameter.device
    return fallback


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move tensor batch values onto the model device when supported."""
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device) if hasattr(value, "to") else value
    return result


def _resolve_classifier_id2label(model: Any) -> dict[int, str]:
    """Resolve classifier id2label mapping from model config."""
    config = getattr(model, "config", None)
    id2label = getattr(config, "id2label", None)
    if not isinstance(id2label, dict) or not id2label:
        raise ValueError(
            "Sequence-classification inference requires model.config.id2label."
        )
    return {int(key): str(value) for key, value in id2label.items()}


def generate_predictions(
    cfg: Config,
    model: Any,
    tokenizer: Any,
    texts: list[str],
) -> list[str]:
    """Generate model predictions for a list of inference texts."""
    if not texts:
        return []

    if hasattr(model, "eval"):
        model.eval()

    device = _resolve_model_device(model, cfg.device)
    batch_size = max(1, cfg.per_device_eval_batch_size)
    raw_predictions: list[str] = []

    for batch_texts in _batched_texts(texts, batch_size=batch_size):
        tokenized_batch = tokenizer(
            batch_texts,
            max_length=cfg.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = _move_batch_to_device(tokenized_batch, device)

        with torch.no_grad():
            if cfg.model_task == "sequence_classification":
                outputs = model(**model_inputs)
                logits = outputs.logits
                predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
                id2label = _resolve_classifier_id2label(model)
                raw_predictions.extend(
                    id2label[int(label_id)] for label_id in predicted_ids
                )
            else:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=cfg.resolved_max_target_length(),
                )
                decoded_predictions = tokenizer.batch_decode(
                    generated_ids.detach().cpu().tolist(),
                    skip_special_tokens=True,
                )
                raw_predictions.extend(str(text) for text in decoded_predictions)

    if len(raw_predictions) != len(texts):
        raise ValueError(
            "Inference produced a different number of predictions than input rows."
        )
    return raw_predictions
