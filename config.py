from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Config:
    hf_model: str = "google/flan-ul2"
    max_source_length: int = 512
    max_target_length: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    output_dir: str = "./runs"
    seed: int = 42

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    device_map: Optional[str] = "auto"
    load_in_8bit: bool = True
    torch_dtype: Optional[str] = "auto"


config = Config()