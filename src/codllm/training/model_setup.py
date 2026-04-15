from typing import Any
import warnings

import torch

from codllm.config import Config
from codllm.data import DataHandler
from codllm.models import load_base_model
from codllm.runtime.reproducibility import configure_reproducibility


def validate_trainable_model(model: Any) -> None:
    """Ensure current pipeline is not asked to full-finetune a quantized base model."""
    if getattr(model, "is_quantized", False):
        raise ValueError(
            "Quantized model detected for fine-tuning. "
            "Set CODLLM_LOAD_IN_8BIT=0 (or cfg.load_in_8bit=False) "
            "or attach PEFT adapters before training."
        )


def model_uses_trainable_fp16_params(model: Any) -> bool:
    """Return True when any trainable floating-point parameter is float16."""
    if not hasattr(model, "parameters"):
        return False
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if torch.is_floating_point(parameter) and parameter.dtype == torch.float16:
            return True
    return False


def upcast_trainable_fp16_params(model: Any) -> bool:
    """Cast the model to float32 when trainable parameters are float16."""
    if not model_uses_trainable_fp16_params(model):
        return False
    if not hasattr(model, "float"):
        return False
    model.float()
    return True


def initialize_training_components(
    cfg: Config,
    label2id: dict[str, int] | None = None,
    id2label: dict[int, str] | None = None,
) -> tuple[Any, Any, bool]:
    """Load model/tokenizer once and apply training-safety dtype guards."""
    configure_reproducibility(cfg)
    model, tokenizer = load_base_model(cfg, label2id=label2id, id2label=id2label)
    validate_trainable_model(model)
    upcasted_fp16_model = upcast_trainable_fp16_params(model)
    disable_fp16 = model_uses_trainable_fp16_params(model)
    if upcasted_fp16_model:
        warnings.warn(
            (
                "Trainable model parameters were loaded as float16. "
                "Upcasting model to float32 to improve optimization stability."
            ),
            stacklevel=2,
        )
    if disable_fp16:
        warnings.warn(
            (
                "Trainable model parameters are already float16. "
                "Disabling Trainer fp16 AMP to avoid grad unscale errors."
            ),
            stacklevel=2,
        )
    return model, tokenizer, disable_fp16


def resolve_classifier_label_space(
    handler: DataHandler,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build classifier label mappings from the configured ICD10h masterlist."""
    labels = handler.get_masterlist_label_vocabulary()
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
