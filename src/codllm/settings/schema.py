from dataclasses import dataclass, field
from typing import Optional

import torch

from codllm.settings.factories import (
    default_base_perturbations,
    default_balance_perturbations,
    default_data_sources,
    default_device,
    default_device_map,
    default_input_field_prefixes,
    default_masterlist_inject_perturbations,
    default_multicod_synthetic_text_separators,
    default_pretrain_perturbations,
    default_pretrain_multicod_synthetic_text_separators,
    default_training_input,
)
from codllm.settings.types import (
    BalanceStrategy,
    EvalStrategy,
    HoldOutEvaluatePer,
    LRSchedulerType,
    ModelTask,
    MultiCodSyntheticSourceScope,
    SaveStrategy,
    SaveStrategyBestMetric,
    TorchDType,
    TrainingInput,
    WandbLogModel,
    WandbMetricMode,
    WandbMode,
    WandbRunConfigMode,
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
    entity: Optional[str] = "codllmdev"
    run_name: Optional[str] = None
    mode: WandbMode = "auto"
    log_model: WandbLogModel = "end"
    run_config_mode: WandbRunConfigMode = "standard"
    metric_mode: WandbMetricMode = "standard"


@dataclass
class Config:
    """Configuration for Hugging Face training experiments."""

    hf_model: str = "google/flan-t5-base"
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
    max_grad_norm: float = 0.5
    warmup_ratio: float = 0.1
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 2
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 5000
    save_total_limit: int | None = None
    eval_strategy: EvalStrategy = "epoch"
    save_strategy: SaveStrategy = "epoch"
    save_strategy_best_metric: SaveStrategyBestMetric = "sample_f1"
    model_task: ModelTask = "seq2seq"
    lr_scheduler_type: LRSchedulerType = "cosine"
    verbose: bool = False
    output_dir: str = "./runs"
    seed: int = 42
    data_seed: Optional[int] = 333
    dataset_sample_seed: Optional[int] = None
    evaluation_protocol: str = "row"
    publication_eval_enabled: bool = False
    evaluation_language_metadata_path: str = "data/curation/source_languages.toml"
    evaluation_language_overrides_path: str | None = None
    evaluation_drop_missing_cod: bool = False
    prediction_export_enabled: bool = False
    publication_gate: str | None = None
    publication_decisions_path: str = "runs/publication/decisions.json"
    evaluation_checkpoint: str | None = None
    evaluation_reference_dir: str | None = None
    evaluation_data_path: str | None = None
    evaluation_scope: str = "external"
    publication_baseline: str = "lookup"
    baseline_max_features: int = 100000
    baseline_max_iter: int = 25
    baseline_alpha: float = 1e-5
    baseline_thresholds: list[float] = field(default_factory=lambda: [0.2, 0.35, 0.5, 0.65])
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
    label_harmonization_enabled: bool = True
    label_harmonization_masterlist_path: str = "data/raw/ICD10h_Masterlist_2024.xlsx"
    label_harmonization_masterlist_sheet_name: str = "Masterlist"
    label_harmonization_transfer_sheet_name: str = "2020to2024transfer"
    label_standardization_enabled: bool = True
    label_standardization_rules_path: str = "data/curation/label_standardization.toml"
    label_standardization_overrides_path: str = (
        "data/curation/label_standardization_overrides.csv"
    )
    hold_out_dataset: Optional[str] = None
    train_excluded_source_ids: list[str] = field(default_factory=list)
    hold_out_evaluate_per: Optional[HoldOutEvaluatePer] = None
    hold_out_evaluate_ratio: float = 0.05
    training_input: list[TrainingInput] = field(default_factory=default_training_input)
    input_field_prefixes: dict[TrainingInput, str] = field(
        default_factory=default_input_field_prefixes
    )
    max_label_count: int = 3
    multicod_shuffle_labels: bool = True
    multicod_synthetic_ratio: float = 0.0
    multicod_synthetic_source_scope: MultiCodSyntheticSourceScope = "within_source"
    multicod_synthetic_text_separators: list[str] = field(
        default_factory=default_multicod_synthetic_text_separators
    )
    inference_validate_registry: bool = False
    final_test_eval_enabled: bool = True
    uncertainty_eval: bool = True

    auto_resume: bool = False
    per_size_output_dir: bool = False
    # On resume, skip HuggingFace's replay of the dataloader up to the saved
    # step. That replay is O(steps already done) and grows every resubmission,
    # so for wall-time-budget campaigns it can waste an hour+ per slot with the
    # GPU idle. Ignoring it restarts the interrupted epoch from a fresh shuffle
    # (optimizer/scheduler/step count still restored) - the right trade for
    # long multi-slot runs.
    ignore_data_skip: bool = True
    load_best_model_at_end: bool = False
    early_stopping_patience: int = 0
    early_stopping_threshold: float = 0.0
    max_runtime_seconds: int = 0
    runtime_safety_margin_seconds: int = 300

    dataset_size: float = 0.5
    train_sample_fraction: float = 1.0
    train_size: float = 0.9
    val_size: float = 0.05
    test_size: float = 0.05
    pretrain_enabled: bool = False
    pretrain_masterlist_path: str = "data/raw/ICD10h_Masterlist_2024.xlsx"
    pretrain_masterlist_sheet_name: str = "Masterlist"
    pretrain_transfer_sheet_name: str = "2020to2024transfer"
    pretrain_num_train_epochs: int = 1
    pretrain_learning_rate: float | None = None
    pretrain_warmup_ratio: float = 0.0
    pretrain_eval_every_n_epochs: int = 1
    pretrain_lr_scheduler_type: LRSchedulerType = "linear"
    pretrain_early_stopping_patience: int | None = None
    pretrain_load_best_model_at_end: bool | None = None
    pretrain_upsample_enabled: bool = True
    pretrain_upsample_target_per_label: int = 10
    pretrain_upsample_perturbations: list[str] = field(
        default_factory=default_pretrain_perturbations
    )
    pretrain_upsample_perturbations_per_sample: int = 1
    pretrain_multicod_synthetic_ratio: float = 0.0
    pretrain_multicod_synthetic_text_separators: list[str] = field(
        default_factory=default_pretrain_multicod_synthetic_text_separators
    )
    masterlist_inject_enabled: bool = False
    masterlist_inject_target_per_label: int = 10
    masterlist_inject_perturbations: list[str] = field(
        default_factory=default_masterlist_inject_perturbations
    )
    masterlist_inject_perturbations_per_sample: int = 1

    balance_strategy: BalanceStrategy = "none"
    balance_perturbations: list[str] = field(
        default_factory=default_balance_perturbations
    )
    balance_perturbation_mean: float = 0.05
    balance_perturbation_variance: float = 0.0
    balance_perturbation_loft: float = 3.0
    balance_floor: int = 0
    balance_floor_decay: float = 0.0
    balance_floor_singlecod_only: bool = False
    base_perturbations: list[str] = field(default_factory=default_base_perturbations)
    base_perturbation_mean: float = 0.05
    base_perturbation_variance: float = 0.0
    base_perturbation_loft: float = 3.0
    base_perturbation_rate: float = 0.05

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

    def resolved_dataset_sample_seed(self) -> int:
        """Return the cohort-sampling seed, defaulting to the data seed."""
        if self.dataset_sample_seed is None:
            return self.resolved_data_seed()
        return self.dataset_sample_seed

    def resolved_pretrain_early_stopping_patience(self) -> int:
        """Return pretraining patience, inheriting fine-tuning when unset."""
        if self.pretrain_early_stopping_patience is None:
            return self.early_stopping_patience
        return self.pretrain_early_stopping_patience

    def resolved_pretrain_load_best_model_at_end(self) -> bool:
        """Return pretraining best-model loading, inheriting when unset."""
        if self.pretrain_load_best_model_at_end is None:
            return self.load_best_model_at_end
        return self.pretrain_load_best_model_at_end

    def input_field_prefix(self, feature: TrainingInput) -> str:
        """Return the configured text prefix for one training input field."""
        prefix = self.input_field_prefixes.get(feature)
        if prefix is None:
            raise KeyError(
                f"Missing input prefix for training input field '{feature}'."
            )
        if prefix == "":
            raise ValueError(
                f"Input prefix for training input field '{feature}' must not be empty."
            )
        return prefix

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
