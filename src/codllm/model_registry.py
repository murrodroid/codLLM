import os
from typing import Callable, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from codllm.config import Config

Loader = Callable[[Config], Tuple[PreTrainedModel, PreTrainedTokenizerBase]]

MODEL_REGISTRY: Dict[str, Loader] = {}


def _apply_hf_runtime_env(cfg: Config) -> None:
    """Set Hugging Face runtime flags required for stable model loading."""
    if cfg.disable_safetensors_conversion:
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"


def _resolve_hf_token(cfg: Config) -> Optional[str]:
    """Resolve Hugging Face token from config, then environment variables."""
    if cfg.hf_token:
        return cfg.hf_token

    return os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")


def _resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[object]:
    """Convert a config dtype string into a torch dtype accepted by transformers."""
    if dtype_name in (None, "auto"):
        return dtype_name

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype '{dtype_name}'.")
    return dtype_map[dtype_name]


def _build_model_kwargs(cfg: Config, token: Optional[str]) -> Dict[str, object]:
    """Build model loading kwargs with hardware-safe defaults."""
    kwargs: Dict[str, object] = {
        "trust_remote_code": cfg.trust_remote_code,
        "use_safetensors": cfg.use_safetensors,
        "torch_dtype": _resolve_torch_dtype(cfg.torch_dtype),
    }

    if cfg.device_map is not None:
        kwargs["device_map"] = cfg.device_map

    if token:
        kwargs["token"] = token

    if cfg.load_in_8bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    return kwargs


def load_default_seq2seq(
    cfg: Config,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load an AutoModelForSeq2SeqLM and tokenizer for any HF model id."""
    _apply_hf_runtime_env(cfg)
    token = _resolve_hf_token(cfg)
    model_kwargs = _build_model_kwargs(cfg, token)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.hf_model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.hf_model,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
        token=token,
    )
    return model, tokenizer


def load_base_model(cfg: Config) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model using a registry override or the default seq2seq loader."""
    loader = MODEL_REGISTRY.get(cfg.hf_model, load_default_seq2seq)
    return loader(cfg)
