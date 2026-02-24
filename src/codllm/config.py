from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


TrainingInput = Literal["cod", "age", "sex"]
WandbMode = Literal["auto", "online", "offline", "disabled"]


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
class DataSourceConfig:
    """Configuration for a single raw dataset source."""

    source_id: str
    path: str
    mapping_id: str
    enabled: bool = True
    file_type: Optional[str] = None
    sep: str = ","
    header: int | None = 0
    sheet_name: int | str = 0
    skip_rows: list[int] = field(default_factory=list)


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases experiment tracking."""

    enabled: bool = True
    project: str = "codllm"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    mode: WandbMode = "auto"


def _default_data_sources() -> list[DataSourceConfig]:
    """Return default data sources expected in the raw data directory."""
    return [
        DataSourceConfig(
            source_id="belgium_1920_1930",
            path="SOSA_EXTR_1920-1930 (belgium).xlsx",
            mapping_id="belgium",
        ),
        DataSourceConfig(
            source_id="amsterdam_1854_1926",
            path="AMC_1854_1926_LM.csv",
            mapping_id="amsterdam",
            sep=";",
        ),
    ]


@dataclass
class Config:
    """Configuration for Hugging Face seq2seq training experiments."""

    hf_model: str = "google/flan-t5-small"  # google/flan-ul2, google/flan-t5-small
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
    use_safetensors: bool = False
    disable_safetensors_conversion: bool = True
    torch_dtype: Optional[str] = "auto"

    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    processed_filename: str = "data.parquet"
    data_sources: list[DataSourceConfig] = field(default_factory=_default_data_sources)
    training_input: list[TrainingInput] = field(
        default_factory=lambda: ["cod", "age", "sex"]
    )
    max_label_count: int = 1

    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1

    dataset_size: float = 0.05
    wandb: WandbConfig = field(default_factory=WandbConfig)


config = Config()
