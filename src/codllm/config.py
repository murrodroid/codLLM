import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, cast

import torch


TrainingInput = Literal["cod", "age", "sex"]
BalanceStrategy = Literal["none", "upsample", "sqrt"]
WandbMode = Literal["auto", "online", "offline", "disabled"]
WandbLogModel = Literal["false", "end", "checkpoint"]
TorchDType = Literal["auto", "float16", "bfloat16", "float32"]
EvalStrategy = Literal["no", "steps", "epoch"]
SaveStrategy = Literal["no", "steps", "epoch", "best"]
SaveStrategyBestMetric = Literal[
    "loss",
    "accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]


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
        DataSourceConfig(
            source_id="copenhagen_may2025",
            path="Copenhagen_burials_all_May2025.csv",
            mapping_id="copenhagen",
        ),
        DataSourceConfig(
            source_id="ipswich_1871_1911",
            path="Ipswich_deaths_codllm.txt",
            mapping_id="ipswich",
            file_type="csv",
            sep="|",
            encoding="latin-1",
        ),
        DataSourceConfig(
            source_id="madrid_1905_1927",
            path="Madrid 1905_1927.csv",
            mapping_id="madrid",
        ),
        DataSourceConfig(
            source_id="historic_strings_en_2024",
            path="ICD10H_HISTORICSTRINGSENGLISH_2024.2.xlsx",
            mapping_id="historic_strings",
            sheet_name="HistoricstringsEnglish2024 1.1",
        ),
    ]


@dataclass
class Config:
    """Configuration for Hugging Face seq2seq training experiments."""

    DEFAULT_LABEL_SEPARATOR: ClassVar[str] = ","
    DEFAULT_TEXT_FIELD_SEPARATOR: ClassVar[str] = " | "
    DEFAULT_DATASET_TEXT_COLUMN: ClassVar[str] = "text"
    DEFAULT_DATASET_LABEL_COLUMN: ClassVar[str] = "label"
    DEFAULT_DATA_RAW_DIR: ClassVar[str] = "data/raw"
    DEFAULT_DATA_PROCESSED_DIR: ClassVar[str] = "data/processed"
    SUPPORTED_TRAINING_INPUTS: ClassVar[tuple[TrainingInput, ...]] = (
        "cod",
        "age",
        "sex",
    )
    SUPPORTED_SAVE_STRATEGY_BEST_METRICS: ClassVar[
        tuple[SaveStrategyBestMetric, ...]
    ] = (
        "loss",
        "accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )

    hf_model: str = "google/flan-t5-small"  # google/flan-ul2, google/flan-t5-small
    hf_token: Optional[str] = None
    trust_remote_code: bool = False

    max_source_length: int = 256
    max_target_length: int = 32
    label_separator: str = DEFAULT_LABEL_SEPARATOR
    text_field_separator: str = DEFAULT_TEXT_FIELD_SEPARATOR
    label_code_length: int = 7
    max_target_length_buffer: int = 4
    dataset_text_column: str = DEFAULT_DATASET_TEXT_COLUMN
    dataset_label_column: str = DEFAULT_DATASET_LABEL_COLUMN

    lr: float = 1e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 0.1
    warmup_steps: int = 1000
    dataloader_num_workers: int = 4
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 5000
    eval_strategy: EvalStrategy = "epoch"
    save_strategy: SaveStrategy = "epoch"
    save_strategy_best_metric: SaveStrategyBestMetric = "accuracy"
    verbose: bool = False
    output_dir: str = "./runs"
    seed: int = 42
    data_seed: Optional[int] = None
    deterministic_algorithms: bool = True
    deterministic_algorithms_warn_only: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    device: torch.device = field(default_factory=_default_device)
    device_map: Optional[str] = field(default_factory=_default_device_map)
    load_in_8bit: bool = False
    use_safetensors: bool = False
    disable_safetensors_conversion: bool = True
    torch_dtype: Optional[TorchDType] = "auto"

    data_raw_dir: str = DEFAULT_DATA_RAW_DIR
    data_processed_dir: str = DEFAULT_DATA_PROCESSED_DIR
    processed_filename: str = "data.parquet"
    data_sources: list[DataSourceConfig] = field(default_factory=_default_data_sources)
    training_input: list[TrainingInput] = field(
        default_factory=lambda: list(Config.SUPPORTED_TRAINING_INPUTS)
    )
    max_label_count: int = 1

    dataset_size: float = 0.5
    train_size: float = 0.9
    val_size: float = 0.05
    test_size: float = 0.05

    balance_strategy: BalanceStrategy = "sqrt"
    balance_target_quantile: float = 0.5
    balance_perturbations: list[str] = field(
        default_factory=lambda: [
            "swap_adjacent_chars",
            "delete_random_char",
            "accent_random_vowel",
            "qwerty_misspell",
        ]
    )
    balance_perturbations_per_sample: int = 1
    balance_upsample_labels: list[str] = field(default_factory=list)
    balance_upsample_inverse_power: float = 0.5
    balance_upsample_budget_ratio: float = 0.4
    balance_sqrt_floor: int = 10
    balance_sqrt_decay: float = 0.1
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


def _parse_training_input(raw_value: str) -> list[TrainingInput]:
    """Parse comma-separated training_input env values."""
    allowed_inputs = set(Config.SUPPORTED_TRAINING_INPUTS)
    parsed: list[TrainingInput] = []
    for feature in raw_value.split(","):
        cleaned_feature = feature.strip().lower()
        if cleaned_feature == "":
            continue
        if cleaned_feature not in allowed_inputs:
            allowed = ", ".join(sorted(allowed_inputs))
            raise ValueError(
                f"CODLLM_TRAINING_INPUT contains unsupported value '{feature}'. "
                f"Supported values are: {allowed}."
            )
        normalized_feature = cast(TrainingInput, cleaned_feature)
        if normalized_feature not in parsed:
            parsed.append(normalized_feature)
    if not parsed:
        raise ValueError("CODLLM_TRAINING_INPUT must include at least one value.")
    return parsed


def config_from_env(base: Optional[Config] = None) -> Config:
    """Create runtime config with environment overrides for training, HPC, and reproducibility."""
    cfg = deepcopy(base) if base is not None else Config()

    hf_model = os.getenv("CODLLM_HF_MODEL")
    if hf_model is not None and hf_model.strip() != "":
        cfg.hf_model = hf_model.strip()

    hf_token = os.getenv("CODLLM_HF_TOKEN")
    if hf_token is not None and hf_token.strip() != "":
        cfg.hf_token = hf_token.strip()

    trust_remote_code = _parse_env_bool("CODLLM_TRUST_REMOTE_CODE")
    if trust_remote_code is not None:
        cfg.trust_remote_code = trust_remote_code

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

    warmup_steps = _parse_env_int("CODLLM_WARMUP_STEPS")
    if warmup_steps is not None:
        if warmup_steps < 0:
            raise ValueError("CODLLM_WARMUP_STEPS must be non-negative.")
        cfg.warmup_steps = warmup_steps

    num_train_epochs = _parse_env_int("CODLLM_NUM_TRAIN_EPOCHS")
    if num_train_epochs is not None:
        if num_train_epochs < 1:
            raise ValueError("CODLLM_NUM_TRAIN_EPOCHS must be at least 1.")
        cfg.num_train_epochs = num_train_epochs

    per_device_train_batch_size = _parse_env_int("CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE")
    if per_device_train_batch_size is not None:
        if per_device_train_batch_size < 1:
            raise ValueError("CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE must be at least 1.")
        cfg.per_device_train_batch_size = per_device_train_batch_size

    per_device_eval_batch_size = _parse_env_int("CODLLM_PER_DEVICE_EVAL_BATCH_SIZE")
    if per_device_eval_batch_size is not None:
        if per_device_eval_batch_size < 1:
            raise ValueError("CODLLM_PER_DEVICE_EVAL_BATCH_SIZE must be at least 1.")
        cfg.per_device_eval_batch_size = per_device_eval_batch_size

    gradient_accumulation_steps = _parse_env_int("CODLLM_GRADIENT_ACCUMULATION_STEPS")
    if gradient_accumulation_steps is not None:
        if gradient_accumulation_steps < 1:
            raise ValueError("CODLLM_GRADIENT_ACCUMULATION_STEPS must be at least 1.")
        cfg.gradient_accumulation_steps = gradient_accumulation_steps

    logging_steps = _parse_env_int("CODLLM_LOGGING_STEPS")
    if logging_steps is not None:
        if logging_steps < 1:
            raise ValueError("CODLLM_LOGGING_STEPS must be at least 1.")
        cfg.logging_steps = logging_steps

    eval_steps = _parse_env_int("CODLLM_EVAL_STEPS")
    if eval_steps is not None:
        if eval_steps < 1:
            raise ValueError("CODLLM_EVAL_STEPS must be at least 1.")
        cfg.eval_steps = eval_steps

    save_steps = _parse_env_int("CODLLM_SAVE_STEPS")
    if save_steps is not None:
        if save_steps < 1:
            raise ValueError("CODLLM_SAVE_STEPS must be at least 1.")
        cfg.save_steps = save_steps

    max_label_count = _parse_env_int("CODLLM_MAX_LABEL_COUNT")
    if max_label_count is not None:
        if max_label_count < 1:
            raise ValueError("CODLLM_MAX_LABEL_COUNT must be at least 1.")
        cfg.max_label_count = max_label_count

    max_source_length = _parse_env_int("CODLLM_MAX_SOURCE_LENGTH")
    if max_source_length is not None:
        if max_source_length < 1:
            raise ValueError("CODLLM_MAX_SOURCE_LENGTH must be at least 1.")
        cfg.max_source_length = max_source_length

    max_target_length = _parse_env_int("CODLLM_MAX_TARGET_LENGTH")
    if max_target_length is not None:
        if max_target_length < 1:
            raise ValueError("CODLLM_MAX_TARGET_LENGTH must be at least 1.")
        cfg.max_target_length = max_target_length

    label_code_length = _parse_env_int("CODLLM_LABEL_CODE_LENGTH")
    if label_code_length is not None:
        if label_code_length < 1:
            raise ValueError("CODLLM_LABEL_CODE_LENGTH must be at least 1.")
        cfg.label_code_length = label_code_length

    max_target_length_buffer = _parse_env_int("CODLLM_MAX_TARGET_LENGTH_BUFFER")
    if max_target_length_buffer is not None:
        if max_target_length_buffer < 0:
            raise ValueError("CODLLM_MAX_TARGET_LENGTH_BUFFER must be non-negative.")
        cfg.max_target_length_buffer = max_target_length_buffer

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

    load_in_8bit = _parse_env_bool("CODLLM_LOAD_IN_8BIT")
    if load_in_8bit is not None:
        cfg.load_in_8bit = load_in_8bit

    use_safetensors = _parse_env_bool("CODLLM_USE_SAFETENSORS")
    if use_safetensors is not None:
        cfg.use_safetensors = use_safetensors

    disable_safetensors_conversion = _parse_env_bool(
        "CODLLM_DISABLE_SAFETENSORS_CONVERSION"
    )
    if disable_safetensors_conversion is not None:
        cfg.disable_safetensors_conversion = disable_safetensors_conversion

    verbose = _parse_env_bool("CODLLM_VERBOSE")
    if verbose is not None:
        cfg.verbose = verbose

    torch_dtype = os.getenv("CODLLM_TORCH_DTYPE")
    if torch_dtype is not None and torch_dtype.strip() != "":
        normalized_torch_dtype = torch_dtype.strip().lower()
        allowed_torch_dtypes = {"auto", "float16", "bfloat16", "float32"}
        if normalized_torch_dtype not in allowed_torch_dtypes:
            allowed = ", ".join(sorted(allowed_torch_dtypes))
            raise ValueError(f"CODLLM_TORCH_DTYPE must be one of: {allowed}.")
        cfg.torch_dtype = cast(TorchDType, normalized_torch_dtype)

    eval_strategy = os.getenv("CODLLM_EVAL_STRATEGY")
    if eval_strategy is not None and eval_strategy.strip() != "":
        normalized_eval_strategy = eval_strategy.strip().lower()
        allowed_eval_strategies = {"no", "steps", "epoch"}
        if normalized_eval_strategy not in allowed_eval_strategies:
            allowed = ", ".join(sorted(allowed_eval_strategies))
            raise ValueError(f"CODLLM_EVAL_STRATEGY must be one of: {allowed}.")
        cfg.eval_strategy = cast(EvalStrategy, normalized_eval_strategy)

    save_strategy = os.getenv("CODLLM_SAVE_STRATEGY")
    if save_strategy is not None and save_strategy.strip() != "":
        normalized_save_strategy = save_strategy.strip().lower()
        allowed_save_strategies = {"no", "steps", "epoch", "best"}
        if normalized_save_strategy not in allowed_save_strategies:
            allowed = ", ".join(sorted(allowed_save_strategies))
            raise ValueError(f"CODLLM_SAVE_STRATEGY must be one of: {allowed}.")
        cfg.save_strategy = cast(SaveStrategy, normalized_save_strategy)

    save_strategy_best_metric = os.getenv("CODLLM_SAVE_STRATEGY_BEST_METRIC")
    if (
        save_strategy_best_metric is not None
        and save_strategy_best_metric.strip() != ""
    ):
        normalized_best_metric = save_strategy_best_metric.strip().lower()
        allowed_best_metrics = set(Config.SUPPORTED_SAVE_STRATEGY_BEST_METRICS)
        if normalized_best_metric not in allowed_best_metrics:
            allowed = ", ".join(sorted(allowed_best_metrics))
            raise ValueError(
                f"CODLLM_SAVE_STRATEGY_BEST_METRIC must be one of: {allowed}."
            )
        cfg.save_strategy_best_metric = cast(
            SaveStrategyBestMetric, normalized_best_metric
        )

    dataset_size = _parse_env_float("CODLLM_DATASET_SIZE")
    if dataset_size is not None:
        cfg.dataset_size = dataset_size

    lr = _parse_env_float("CODLLM_LR")
    if lr is not None:
        if lr <= 0:
            raise ValueError("CODLLM_LR must be positive.")
        cfg.lr = lr

    max_grad_norm = _parse_env_float("CODLLM_MAX_GRAD_NORM")
    if max_grad_norm is not None:
        if max_grad_norm < 0:
            raise ValueError("CODLLM_MAX_GRAD_NORM must be non-negative.")
        cfg.max_grad_norm = max_grad_norm

    weight_decay = _parse_env_float("CODLLM_WEIGHT_DECAY")
    if weight_decay is not None:
        if weight_decay < 0:
            raise ValueError("CODLLM_WEIGHT_DECAY must be non-negative.")
        cfg.weight_decay = weight_decay

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

    label_separator = os.getenv("CODLLM_LABEL_SEPARATOR")
    if label_separator is not None:
        cfg.label_separator = label_separator

    text_field_separator = os.getenv("CODLLM_TEXT_FIELD_SEPARATOR")
    if text_field_separator is not None:
        cfg.text_field_separator = text_field_separator

    dataset_text_column = os.getenv("CODLLM_DATASET_TEXT_COLUMN")
    if dataset_text_column is not None and dataset_text_column.strip() != "":
        cfg.dataset_text_column = dataset_text_column.strip()

    dataset_label_column = os.getenv("CODLLM_DATASET_LABEL_COLUMN")
    if dataset_label_column is not None and dataset_label_column.strip() != "":
        cfg.dataset_label_column = dataset_label_column.strip()

    device = os.getenv("CODLLM_DEVICE")
    if device is not None and device.strip() != "":
        normalized_device = device.strip().lower()
        allowed_devices = {"cpu", "cuda", "mps"}
        if normalized_device not in allowed_devices:
            allowed = ", ".join(sorted(allowed_devices))
            raise ValueError(f"CODLLM_DEVICE must be one of: {allowed}.")
        cfg.device = torch.device(normalized_device)

    device_map = os.getenv("CODLLM_DEVICE_MAP")
    if device_map is not None and device_map.strip() != "":
        normalized_device_map = device_map.strip().lower()
        if normalized_device_map in {"none", "null", "off"}:
            cfg.device_map = None
        else:
            cfg.device_map = device_map.strip()

    training_input = os.getenv("CODLLM_TRAINING_INPUT")
    if training_input is not None and training_input.strip() != "":
        cfg.training_input = _parse_training_input(training_input)

    data_raw_dir = os.getenv("CODLLM_DATA_RAW_DIR")
    if data_raw_dir:
        cfg.data_raw_dir = data_raw_dir

    data_processed_dir = os.getenv("CODLLM_DATA_PROCESSED_DIR")
    if data_processed_dir:
        cfg.data_processed_dir = data_processed_dir

    balance_strategy = os.getenv("CODLLM_BALANCE_STRATEGY")
    if balance_strategy is not None and balance_strategy.strip() != "":
        normalized_balance = balance_strategy.strip().lower()
        allowed_balance = {"none", "upsample", "sqrt"}
        if normalized_balance not in allowed_balance:
            allowed = ", ".join(sorted(allowed_balance))
            raise ValueError(f"CODLLM_BALANCE_STRATEGY must be one of: {allowed}.")
        cfg.balance_strategy = cast(BalanceStrategy, normalized_balance)

    balance_target_quantile = _parse_env_float("CODLLM_BALANCE_TARGET_QUANTILE")
    if balance_target_quantile is not None:
        if balance_target_quantile < 0 or balance_target_quantile > 1:
            raise ValueError("CODLLM_BALANCE_TARGET_QUANTILE must be between 0 and 1.")
        cfg.balance_target_quantile = balance_target_quantile

    balance_perturbations = os.getenv("CODLLM_BALANCE_PERTURBATIONS")
    if balance_perturbations is not None and balance_perturbations.strip() != "":
        cfg.balance_perturbations = [
            p.strip() for p in balance_perturbations.split(",") if p.strip()
        ]

    balance_perturbations_per_sample = _parse_env_int(
        "CODLLM_BALANCE_PERTURBATIONS_PER_SAMPLE"
    )
    if balance_perturbations_per_sample is not None:
        if balance_perturbations_per_sample < 1:
            raise ValueError(
                "CODLLM_BALANCE_PERTURBATIONS_PER_SAMPLE must be at least 1."
            )
        cfg.balance_perturbations_per_sample = balance_perturbations_per_sample

    balance_upsample_labels = os.getenv("CODLLM_BALANCE_UPSAMPLE_LABELS")
    if balance_upsample_labels is not None and balance_upsample_labels.strip() != "":
        cfg.balance_upsample_labels = [
            label.strip()
            for label in balance_upsample_labels.split(",")
            if label.strip()
        ]

    balance_upsample_inverse_power = _parse_env_float(
        "CODLLM_BALANCE_UPSAMPLE_INVERSE_POWER"
    )
    if balance_upsample_inverse_power is not None:
        if balance_upsample_inverse_power <= 0 or balance_upsample_inverse_power > 1:
            raise ValueError(
                "CODLLM_BALANCE_UPSAMPLE_INVERSE_POWER must be in the interval (0, 1]."
            )
        cfg.balance_upsample_inverse_power = balance_upsample_inverse_power

    balance_upsample_budget_ratio = _parse_env_float(
        "CODLLM_BALANCE_UPSAMPLE_BUDGET_RATIO"
    )
    if balance_upsample_budget_ratio is not None:
        if balance_upsample_budget_ratio < 0 or balance_upsample_budget_ratio > 1:
            raise ValueError(
                "CODLLM_BALANCE_UPSAMPLE_BUDGET_RATIO must be between 0 and 1."
            )
        cfg.balance_upsample_budget_ratio = balance_upsample_budget_ratio

    balance_sqrt_floor = _parse_env_int("CODLLM_BALANCE_SQRT_FLOOR")
    if balance_sqrt_floor is not None:
        if balance_sqrt_floor < 1:
            raise ValueError("CODLLM_BALANCE_SQRT_FLOOR must be at least 1.")
        cfg.balance_sqrt_floor = balance_sqrt_floor

    balance_sqrt_decay = _parse_env_float("CODLLM_BALANCE_SQRT_DECAY")
    if balance_sqrt_decay is not None:
        if balance_sqrt_decay < 0 or balance_sqrt_decay > 1:
            raise ValueError("CODLLM_BALANCE_SQRT_DECAY must be between 0 and 1.")
        cfg.balance_sqrt_decay = balance_sqrt_decay

    balance_sqrt_power = _parse_env_float("CODLLM_BALANCE_SQRT_POWER")
    if balance_sqrt_power is not None:
        if balance_sqrt_power < 0 or balance_sqrt_power > 1:
            raise ValueError("CODLLM_BALANCE_SQRT_POWER must be between 0 and 1.")
        cfg.balance_sqrt_power = balance_sqrt_power

    balance_sqrt_budget_scale = _parse_env_float("CODLLM_BALANCE_SQRT_BUDGET_SCALE")
    if balance_sqrt_budget_scale is not None:
        if balance_sqrt_budget_scale < 1:
            raise ValueError("CODLLM_BALANCE_SQRT_BUDGET_SCALE must be at least 1.")
        cfg.balance_sqrt_budget_scale = balance_sqrt_budget_scale

    balance_base_perturbation_rate = _parse_env_float(
        "CODLLM_BALANCE_BASE_PERTURBATION_RATE"
    )
    if balance_base_perturbation_rate is not None:
        if balance_base_perturbation_rate < 0 or balance_base_perturbation_rate > 1:
            raise ValueError(
                "CODLLM_BALANCE_BASE_PERTURBATION_RATE must be between 0 and 1."
            )
        cfg.balance_base_perturbation_rate = balance_base_perturbation_rate

    wandb_log_model = os.getenv("CODLLM_WANDB_LOG_MODEL")
    if wandb_log_model is not None and wandb_log_model.strip() != "":
        normalized_wandb_log_model = wandb_log_model.strip().lower()
        allowed_wandb_log_models = {"false", "end", "checkpoint"}
        if normalized_wandb_log_model not in allowed_wandb_log_models:
            allowed = ", ".join(sorted(allowed_wandb_log_models))
            raise ValueError(f"CODLLM_WANDB_LOG_MODEL must be one of: {allowed}.")
        cfg.wandb.log_model = cast(WandbLogModel, normalized_wandb_log_model)

    wandb_enabled = _parse_env_bool("CODLLM_WANDB_ENABLED")
    if wandb_enabled is not None:
        cfg.wandb.enabled = wandb_enabled

    wandb_mode = os.getenv("CODLLM_WANDB_MODE")
    if wandb_mode is not None and wandb_mode.strip() != "":
        normalized_wandb_mode = wandb_mode.strip().lower()
        allowed_wandb_modes = {"auto", "online", "offline", "disabled"}
        if normalized_wandb_mode not in allowed_wandb_modes:
            allowed = ", ".join(sorted(allowed_wandb_modes))
            raise ValueError(f"CODLLM_WANDB_MODE must be one of: {allowed}.")
        cfg.wandb.mode = cast(WandbMode, normalized_wandb_mode)

    wandb_project = os.getenv("CODLLM_WANDB_PROJECT")
    if wandb_project is not None and wandb_project.strip() != "":
        cfg.wandb.project = wandb_project.strip()

    wandb_entity = os.getenv("CODLLM_WANDB_ENTITY")
    if wandb_entity is not None and wandb_entity.strip() != "":
        cfg.wandb.entity = wandb_entity.strip()

    wandb_run_name = os.getenv("CODLLM_WANDB_RUN_NAME")
    if wandb_run_name is not None and wandb_run_name.strip() != "":
        cfg.wandb.run_name = wandb_run_name.strip()

    return cfg


config = Config()
