import os
from copy import deepcopy
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
    data_seed: Optional[int] = None
    deterministic_algorithms: bool = True
    deterministic_algorithms_warn_only: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    dataloader_num_workers: int = 0

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

    dataset_size: float = 0.05
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1

    wandb: WandbConfig = field(default_factory=WandbConfig)


def _parse_env_int(name: str) -> Optional[int]:
    """Parse an optional integer environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable '{name}' must be an integer.") from exc


def _parse_env_bool(name: str) -> Optional[bool]:
    """Parse an optional boolean environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable '{name}' must be one of 1/0, true/false, yes/no, on/off."
    )


def _parse_env_float(name: str) -> Optional[float]:
    """Parse an optional float environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable '{name}' must be a float.") from exc


def config_from_env(base: Optional[Config] = None) -> Config:
    """Create runtime config with environment overrides for HPC and reproducibility."""
    cfg = deepcopy(base) if base is not None else Config()

    seed = _parse_env_int("CODLLM_SEED")
    if seed is not None:
        cfg.seed = seed

    data_seed = _parse_env_int("CODLLM_DATA_SEED")
    if data_seed is not None:
        cfg.data_seed = data_seed

    dataloader_num_workers = _parse_env_int("CODLLM_DATALOADER_NUM_WORKERS")
    if dataloader_num_workers is not None:
        if dataloader_num_workers < 0:
            raise ValueError("CODLLM_DATALOADER_NUM_WORKERS must be non-negative.")
        cfg.dataloader_num_workers = dataloader_num_workers

    deterministic_algorithms = _parse_env_bool("CODLLM_DETERMINISTIC_ALGORITHMS")
    if deterministic_algorithms is not None:
        cfg.deterministic_algorithms = deterministic_algorithms

    deterministic_warn_only = _parse_env_bool(
        "CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY"
    )
    if deterministic_warn_only is not None:
        cfg.deterministic_algorithms_warn_only = deterministic_warn_only

    cudnn_deterministic = _parse_env_bool("CODLLM_CUDNN_DETERMINISTIC")
    if cudnn_deterministic is not None:
        cfg.cudnn_deterministic = cudnn_deterministic

    cudnn_benchmark = _parse_env_bool("CODLLM_CUDNN_BENCHMARK")
    if cudnn_benchmark is not None:
        cfg.cudnn_benchmark = cudnn_benchmark

    dataset_size = _parse_env_float("CODLLM_DATASET_SIZE")
    if dataset_size is not None:
        cfg.dataset_size = dataset_size

    train_size = _parse_env_float("CODLLM_TRAIN_SIZE")
    if train_size is not None:
        cfg.train_size = train_size

    val_size = _parse_env_float("CODLLM_VAL_SIZE")
    if val_size is not None:
        cfg.val_size = val_size

    test_size = _parse_env_float("CODLLM_TEST_SIZE")
    if test_size is not None:
        cfg.test_size = test_size

    output_dir = os.getenv("CODLLM_OUTPUT_DIR")
    if output_dir:
        cfg.output_dir = output_dir

    data_raw_dir = os.getenv("CODLLM_DATA_RAW_DIR")
    if data_raw_dir:
        cfg.data_raw_dir = data_raw_dir

    data_processed_dir = os.getenv("CODLLM_DATA_PROCESSED_DIR")
    if data_processed_dir:
        cfg.data_processed_dir = data_processed_dir

    return cfg


config = Config()
