from dataclasses import dataclass, field
from typing import Optional

import torch

from codllm.settings.factories import (
    default_balance_perturbations,
    default_data_sources,
    default_device,
    default_device_map,
    default_masterlist_inject_perturbations,
    default_pretrain_perturbations,
    default_training_input,
)
from codllm.settings.types import (
    BalanceStrategy,
    EvalStrategy,
    LRSchedulerType,
    ModelTask,
    SaveStrategy,
    SaveStrategyBestMetric,
    TorchDType,
    TrainingInput,
    WandbLogModel,
    WandbMode,
)


@dataclass
class DataSourceConfig:
    """Configuration for a single raw dataset source."""

    source_id: str
    path: str
    mapping_id: str
    enabled: bool = True
    file_type: Optional[str] = None
    sep: str = ","
    encoding: str = "utf-8"
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
    log_model: WandbLogModel = "end"


@dataclass
class Config:
    """Configuration for Hugging Face training experiments."""

    hf_model: str = "google/flan-t5-small"
    hf_token: Optional[str] = None
    trust_remote_code: bool = False

    max_source_length: int = 256
    max_target_length: int = 32
    label_separator: str = ","
    text_field_separator: str = " | "
    label_code_length: int = 7
    max_target_length_buffer: int = 4
    dataset_text_column: str = "text"
    dataset_label_column: str = "label"

    lr: float = 1e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 0.1
    warmup_steps: int = 1000
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 2
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 5000
    eval_strategy: EvalStrategy = "epoch"
    save_strategy: SaveStrategy = "epoch"
    save_strategy_best_metric: SaveStrategyBestMetric = "accuracy"
    model_task: ModelTask = "seq2seq"
    lr_scheduler_type: LRSchedulerType = "linear"
    verbose: bool = False
    output_dir: str = "./runs"
    seed: int = 42
    data_seed: Optional[int] = None
    deterministic_algorithms: bool = True
    deterministic_algorithms_warn_only: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    device: torch.device = field(default_factory=default_device)
    device_map: Optional[str] = field(default_factory=default_device_map)
    load_in_8bit: bool = False
    use_safetensors: bool = False
    disable_safetensors_conversion: bool = True
    torch_dtype: Optional[TorchDType] = "auto"

    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    processed_filename: str = "data.parquet"
    data_sources: list[DataSourceConfig] = field(default_factory=default_data_sources)
    training_input: list[TrainingInput] = field(default_factory=default_training_input)
    max_label_count: int = 1
    inference_validate_registry: bool = False

    dataset_size: float = 0.5
    train_size: float = 0.9
    val_size: float = 0.05
    test_size: float = 0.05
    pretrain_enabled: bool = False
    pretrain_masterlist_path: str = "data/raw/ICD10h_Masterlist_2024.xlsx"
    pretrain_masterlist_sheet_name: str = "Masterlist"
    pretrain_transfer_sheet_name: str = "2020to2024transfer"
    pretrain_num_train_epochs: int = 1
    pretrain_learning_rate: float | None = None
    pretrain_eval_every_n_epochs: int = 1
    pretrain_lr_scheduler_type: LRSchedulerType = "linear"
    pretrain_upsample_enabled: bool = True
    pretrain_upsample_target_per_label: int = 10
    pretrain_upsample_perturbations: list[str] = field(
        default_factory=default_pretrain_perturbations
    )
    pretrain_upsample_perturbations_per_sample: int = 1
    masterlist_inject_enabled: bool = False
    masterlist_inject_target_per_label: int = 10
    masterlist_inject_perturbations: list[str] = field(
        default_factory=default_masterlist_inject_perturbations
    )
    masterlist_inject_perturbations_per_sample: int = 1
    label_harmonization_enabled: bool = False

    balance_strategy: BalanceStrategy = "sqrt"
    balance_target_quantile: float = 0.5
    balance_perturbations: list[str] = field(
        default_factory=default_balance_perturbations
    )
    balance_perturbations_per_sample: int = 1
    balance_upsample_labels: list[str] = field(default_factory=list)
    balance_upsample_inverse_power: float = 0.5
    balance_upsample_budget_ratio: float = 0.4
    balance_sqrt_floor: int = 0
    balance_sqrt_decay: float = 0.0
    balance_sqrt_power: float = 0.5
    balance_sqrt_budget_scale: float = 1.05
    balance_base_perturbation_rate: float = 0.05

    wandb: WandbConfig = field(default_factory=WandbConfig)

    def resolved_max_target_length(self) -> int:
        """Return effective target length using code-format-aware lower bounds."""
        if self.max_label_count < 1:
            raise ValueError("max_label_count must be at least 1.")
        if self.label_code_length < 1:
            raise ValueError("label_code_length must be at least 1.")
        if self.max_target_length_buffer < 0:
            raise ValueError("max_target_length_buffer must be non-negative.")

        separator_count = max(0, self.max_label_count - 1)
        formatted_length = (
            self.max_label_count * self.label_code_length
            + separator_count * len(self.label_separator)
        )
        inferred_min_length = formatted_length + self.max_target_length_buffer
        return max(self.max_target_length, inferred_min_length)

    def resolved_data_seed(self) -> int:
        """Return data seed, defaulting to the global seed when unset."""
        return self.seed if self.data_seed is None else self.data_seed

    def uses_cuda(self) -> bool:
        """Return True when config targets CUDA and CUDA runtime is available."""
        return self.device.type == "cuda" and torch.cuda.is_available()

    def bf16_amp_supported(self) -> bool:
        """Return True when current CUDA runtime supports bf16 AMP."""
        return (
            self.uses_cuda()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )

    def resolved_device_map(self) -> Optional[str]:
        """Return an effective device map compatible with the configured device."""
        if self.device_map == "auto" and self.device.type != "cuda":
            return None
        return self.device_map
