from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from config import Config
from typing import Callable, Dict, Tuple


Loader = Callable[[Config], Tuple[torch.nn.Module, object]]

def load_flan_ul2(cfg: Config):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.hf_model,
        device_map=cfg.device_map,
        load_in_8bit=cfg.load_in_8bit,
        torch_dtype=cfg.torch_dtype,
    )
    tok = AutoTokenizer.from_pretrained(cfg.hf_model, use_fast=True)
    return model, tok

MODEL_REGISTRY: Dict[str, Loader] = {
    "google/flan-ul2": load_flan_ul2,
}

def load_base_model(cfg: Config):
    if cfg.hf_model not in MODEL_REGISTRY:
        raise ValueError(f"Model '{cfg.hf_model}' not registered.")
    return MODEL_REGISTRY[cfg.hf_model](cfg)

