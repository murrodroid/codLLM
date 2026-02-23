from dataclasses import dataclass, field
from typing import Optional

import torch


def _default_device() -> torch.device:
    """Choose the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_device_map() -> Optional[str]:
    """Choose a sensible default device map for the current hardware."""
    if torch.cuda.is_available():
        return "auto"
    return None


@dataclass
class Config:
    """Configuration for Hugging Face seq2seq training experiments."""

    hf_model: str = "google/flan-ul2"
    hf_token: Optional[str] = None
    trust_remote_code: bool = False

    max_source_length: int = 512
    max_target_length: int = 16
    dataset_text_column: str = "text"
    dataset_label_column: str = "label"

    lr: float = 2e-4
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = None
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    output_dir: str = "./runs"
    seed: int = 42

    device: torch.device = field(default_factory=_default_device)
    device_map: Optional[str] = field(default_factory=_default_device_map)
    load_in_8bit: bool = field(default_factory=torch.cuda.is_available)
    torch_dtype: Optional[str] = "auto"

@dataclass
class Data:
    datasets = ['SOSA_EXTR_1920-1930 (belgium).xlsx']
    
    dataset1_X = ""
    dataset1_y = ""

config = Config()
