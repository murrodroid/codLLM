import os
import warnings
from copy import deepcopy
from typing import Optional, cast

import torch

from codllm.settings.options import (
    SUPPORTED_LR_SCHEDULER_TYPES,
    SUPPORTED_MODEL_TASKS,
    SUPPORTED_MULTICOD_SYNTHETIC_SOURCE_SCOPES,
    SUPPORTED_SAVE_STRATEGY_BEST_METRICS,
    SUPPORTED_TRAINING_INPUTS,
)
from codllm.settings.schema import (
    Config,
)
from codllm.settings.types import (
    BalanceStrategy,
    EvalStrategy,
    LRSchedulerType,
    ModelTask,
    MultiCodSyntheticSourceScope,
    HoldOutEvaluatePer,
    SaveStrategy,
    SaveStrategyBestMetric,
    TorchDType,
    TrainingInput,
    WandbLogModel,
    WandbMetricMode,
    WandbMode,
    WandbRunConfigMode,
)


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
    allowed_inputs = set(SUPPORTED_TRAINING_INPUTS)
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


def _apply_input_prefix_env(cfg: Config) -> None:
    """Apply optional processed-input prefix overrides from environment."""
    env_names: dict[TrainingInput, str] = {
        "cod": "CODLLM_INPUT_PREFIX_COD",
        "age": "CODLLM_INPUT_PREFIX_AGE",
        "sex": "CODLLM_INPUT_PREFIX_SEX",
    }
    for feature, env_name in env_names.items():
        prefix = os.getenv(env_name)
        if prefix is None:
            continue
        if prefix == "":
            raise ValueError(f"{env_name} must not be empty.")
        cfg.input_field_prefixes[feature] = prefix


def config_from_env(base: Optional[Config] = None) -> Config:
    """Create runtime config with environment overrides for training and reproducibility."""
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

    dataloader_pin_memory = _parse_env_bool("CODLLM_DATALOADER_PIN_MEMORY")
    if dataloader_pin_memory is not None:
        cfg.dataloader_pin_memory = dataloader_pin_memory

    dataloader_persistent_workers = _parse_env_bool(
        "CODLLM_DATALOADER_PERSISTENT_WORKERS"
    )
    if dataloader_persistent_workers is not None:
        cfg.dataloader_persistent_workers = dataloader_persistent_workers

    dataloader_prefetch_factor = _parse_env_int("CODLLM_DATALOADER_PREFETCH_FACTOR")
    if dataloader_prefetch_factor is not None:
        if dataloader_prefetch_factor < 1:
            raise ValueError("CODLLM_DATALOADER_PREFETCH_FACTOR must be at least 1.")
        cfg.dataloader_prefetch_factor = dataloader_prefetch_factor

    warmup_ratio = _parse_env_float("CODLLM_WARMUP_RATIO")
    if warmup_ratio is not None:
        if warmup_ratio < 0 or warmup_ratio > 1:
            raise ValueError("CODLLM_WARMUP_RATIO must be between 0 and 1.")
        cfg.warmup_ratio = warmup_ratio

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

    multicod_shuffle_labels = _parse_env_bool("CODLLM_MULTICOD_SHUFFLE_LABELS")
    if multicod_shuffle_labels is not None:
        cfg.multicod_shuffle_labels = multicod_shuffle_labels

    multicod_synthetic_ratio = _parse_env_float("CODLLM_MULTICOD_SYNTHETIC_RATIO")
    if multicod_synthetic_ratio is not None:
        if multicod_synthetic_ratio < 0:
            raise ValueError("CODLLM_MULTICOD_SYNTHETIC_RATIO must be non-negative.")
        cfg.multicod_synthetic_ratio = multicod_synthetic_ratio

    multicod_synthetic_source_scope = os.getenv(
        "CODLLM_MULTICOD_SYNTHETIC_SOURCE_SCOPE"
    )
    if (
        multicod_synthetic_source_scope is not None
        and multicod_synthetic_source_scope.strip() != ""
    ):
        normalized_scope = multicod_synthetic_source_scope.strip().lower()
        allowed_scopes = set(SUPPORTED_MULTICOD_SYNTHETIC_SOURCE_SCOPES)
        if normalized_scope not in allowed_scopes:
            allowed = ", ".join(sorted(allowed_scopes))
            raise ValueError(
                f"CODLLM_MULTICOD_SYNTHETIC_SOURCE_SCOPE must be one of: {allowed}."
            )
        cfg.multicod_synthetic_source_scope = cast(
            MultiCodSyntheticSourceScope, normalized_scope
        )

    multicod_synthetic_text_separator = os.getenv(
        "CODLLM_MULTICOD_SYNTHETIC_TEXT_SEPARATOR"
    )
    if multicod_synthetic_text_separator is not None:
        cfg.multicod_synthetic_text_separator = multicod_synthetic_text_separator

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

    uncertainty_eval_enabled = _parse_env_bool("CODLLM_UNCERTAINTY_EVAL_ENABLED")
    if uncertainty_eval_enabled is not None:
        cfg.uncertainty_eval_enabled = uncertainty_eval_enabled

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
        allowed_best_metrics = set(SUPPORTED_SAVE_STRATEGY_BEST_METRICS)
        if normalized_best_metric not in allowed_best_metrics:
            allowed = ", ".join(sorted(allowed_best_metrics))
            raise ValueError(
                f"CODLLM_SAVE_STRATEGY_BEST_METRIC must be one of: {allowed}."
            )
        cfg.save_strategy_best_metric = cast(
            SaveStrategyBestMetric, normalized_best_metric
        )

    model_task = os.getenv("CODLLM_MODEL_TASK")
    if model_task is not None and model_task.strip() != "":
        normalized_model_task = model_task.strip().lower()
        allowed_model_tasks = set(SUPPORTED_MODEL_TASKS)
        if normalized_model_task not in allowed_model_tasks:
            allowed = ", ".join(sorted(allowed_model_tasks))
            raise ValueError(f"CODLLM_MODEL_TASK must be one of: {allowed}.")
        cfg.model_task = cast(ModelTask, normalized_model_task)

    lr_scheduler_type = os.getenv("CODLLM_LR_SCHEDULER_TYPE")
    if lr_scheduler_type is not None and lr_scheduler_type.strip() != "":
        normalized_lr_scheduler_type = lr_scheduler_type.strip().lower()
        allowed_lr_schedulers = set(SUPPORTED_LR_SCHEDULER_TYPES)
        if normalized_lr_scheduler_type not in allowed_lr_schedulers:
            allowed = ", ".join(sorted(allowed_lr_schedulers))
            raise ValueError(f"CODLLM_LR_SCHEDULER_TYPE must be one of: {allowed}.")
        cfg.lr_scheduler_type = cast(LRSchedulerType, normalized_lr_scheduler_type)

    dataset_size = _parse_env_float("CODLLM_DATASET_SIZE")
    if dataset_size is not None:
        cfg.dataset_size = dataset_size

    hold_out_dataset = os.getenv("CODLLM_HOLD_OUT_DATASET")
    if hold_out_dataset is not None:
        normalized_hold_out_dataset = hold_out_dataset.strip()
        cfg.hold_out_dataset = normalized_hold_out_dataset or None

    hold_out_evaluate_per = os.getenv("CODLLM_HOLD_OUT_EVALUATE_PER")
    if hold_out_evaluate_per is not None:
        normalized_hold_out_evaluate_per = hold_out_evaluate_per.strip().lower()
        if normalized_hold_out_evaluate_per in {"", "none", "null", "no", "off"}:
            cfg.hold_out_evaluate_per = None
        else:
            aliases = {
                "step": "steps",
                "steps": "steps",
                "epoch": "epoch",
                "epochs": "epoch",
                "epoche": "epoch",
            }
            if normalized_hold_out_evaluate_per not in aliases:
                allowed = "epoch, steps, none"
                raise ValueError(
                    f"CODLLM_HOLD_OUT_EVALUATE_PER must be one of: {allowed}."
                )
            cfg.hold_out_evaluate_per = cast(
                HoldOutEvaluatePer,
                aliases[normalized_hold_out_evaluate_per],
            )

    hold_out_evaluate_ratio = _parse_env_float("CODLLM_HOLD_OUT_EVALUATE_RATIO")
    if hold_out_evaluate_ratio is not None:
        if hold_out_evaluate_ratio <= 0 or hold_out_evaluate_ratio > 1:
            raise ValueError(
                "CODLLM_HOLD_OUT_EVALUATE_RATIO must be in the interval (0, 1]."
            )
        cfg.hold_out_evaluate_ratio = hold_out_evaluate_ratio

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

    pretrain_enabled = _parse_env_bool("CODLLM_PRETRAIN_ENABLED")
    if pretrain_enabled is not None:
        cfg.pretrain_enabled = pretrain_enabled

    pretrain_masterlist_path = os.getenv("CODLLM_PRETRAIN_MASTERLIST_PATH")
    if pretrain_masterlist_path is not None and pretrain_masterlist_path.strip() != "":
        cfg.pretrain_masterlist_path = pretrain_masterlist_path.strip()

    pretrain_masterlist_sheet_name = os.getenv("CODLLM_PRETRAIN_MASTERLIST_SHEET_NAME")
    if (
        pretrain_masterlist_sheet_name is not None
        and pretrain_masterlist_sheet_name.strip() != ""
    ):
        cfg.pretrain_masterlist_sheet_name = pretrain_masterlist_sheet_name.strip()

    pretrain_transfer_sheet_name = os.getenv("CODLLM_PRETRAIN_TRANSFER_SHEET_NAME")
    if (
        pretrain_transfer_sheet_name is not None
        and pretrain_transfer_sheet_name.strip() != ""
    ):
        cfg.pretrain_transfer_sheet_name = pretrain_transfer_sheet_name.strip()

    label_harmonization_enabled = _parse_env_bool("CODLLM_LABEL_HARMONIZATION_ENABLED")
    if label_harmonization_enabled is not None:
        cfg.label_harmonization_enabled = label_harmonization_enabled

    label_harmonization_masterlist_path = os.getenv(
        "CODLLM_LABEL_HARMONIZATION_MASTERLIST_PATH"
    )
    if (
        label_harmonization_masterlist_path is not None
        and label_harmonization_masterlist_path.strip() != ""
    ):
        cfg.label_harmonization_masterlist_path = (
            label_harmonization_masterlist_path.strip()
        )
    elif (
        pretrain_masterlist_path is not None and pretrain_masterlist_path.strip() != ""
    ):
        cfg.label_harmonization_masterlist_path = cfg.pretrain_masterlist_path

    label_harmonization_masterlist_sheet_name = os.getenv(
        "CODLLM_LABEL_HARMONIZATION_MASTERLIST_SHEET_NAME"
    )
    if (
        label_harmonization_masterlist_sheet_name is not None
        and label_harmonization_masterlist_sheet_name.strip() != ""
    ):
        cfg.label_harmonization_masterlist_sheet_name = (
            label_harmonization_masterlist_sheet_name.strip()
        )
    elif (
        pretrain_masterlist_sheet_name is not None
        and pretrain_masterlist_sheet_name.strip() != ""
    ):
        cfg.label_harmonization_masterlist_sheet_name = (
            cfg.pretrain_masterlist_sheet_name
        )

    label_harmonization_transfer_sheet_name = os.getenv(
        "CODLLM_LABEL_HARMONIZATION_TRANSFER_SHEET_NAME"
    )
    if (
        label_harmonization_transfer_sheet_name is not None
        and label_harmonization_transfer_sheet_name.strip() != ""
    ):
        cfg.label_harmonization_transfer_sheet_name = (
            label_harmonization_transfer_sheet_name.strip()
        )
    elif (
        pretrain_transfer_sheet_name is not None
        and pretrain_transfer_sheet_name.strip() != ""
    ):
        cfg.label_harmonization_transfer_sheet_name = cfg.pretrain_transfer_sheet_name

    pretrain_num_train_epochs = _parse_env_int("CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS")
    if pretrain_num_train_epochs is not None:
        if pretrain_num_train_epochs < 1:
            raise ValueError("CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS must be at least 1.")
        cfg.pretrain_num_train_epochs = pretrain_num_train_epochs

    pretrain_learning_rate = _parse_env_float("CODLLM_PRETRAIN_LEARNING_RATE")
    if pretrain_learning_rate is not None:
        if pretrain_learning_rate <= 0:
            raise ValueError("CODLLM_PRETRAIN_LEARNING_RATE must be positive.")
        cfg.pretrain_learning_rate = pretrain_learning_rate

    pretrain_warmup_ratio = _parse_env_float("CODLLM_PRETRAIN_WARMUP_RATIO")
    if pretrain_warmup_ratio is not None:
        if pretrain_warmup_ratio < 0 or pretrain_warmup_ratio > 1:
            raise ValueError("CODLLM_PRETRAIN_WARMUP_RATIO must be between 0 and 1.")
        cfg.pretrain_warmup_ratio = pretrain_warmup_ratio

    pretrain_eval_every_n_epochs = _parse_env_int("CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS")
    if pretrain_eval_every_n_epochs is not None:
        if pretrain_eval_every_n_epochs < 1:
            raise ValueError("CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS must be at least 1.")
        cfg.pretrain_eval_every_n_epochs = pretrain_eval_every_n_epochs

    pretrain_lr_scheduler_type = os.getenv("CODLLM_PRETRAIN_LR_SCHEDULER_TYPE")
    if (
        pretrain_lr_scheduler_type is not None
        and pretrain_lr_scheduler_type.strip() != ""
    ):
        normalized_pretrain_lr_scheduler_type = (
            pretrain_lr_scheduler_type.strip().lower()
        )
        allowed_lr_schedulers = set(SUPPORTED_LR_SCHEDULER_TYPES)
        if normalized_pretrain_lr_scheduler_type not in allowed_lr_schedulers:
            allowed = ", ".join(sorted(allowed_lr_schedulers))
            raise ValueError(
                f"CODLLM_PRETRAIN_LR_SCHEDULER_TYPE must be one of: {allowed}."
            )
        cfg.pretrain_lr_scheduler_type = cast(
            LRSchedulerType, normalized_pretrain_lr_scheduler_type
        )

    pretrain_upsample_enabled = _parse_env_bool("CODLLM_PRETRAIN_UPSAMPLE_ENABLED")
    if pretrain_upsample_enabled is not None:
        cfg.pretrain_upsample_enabled = pretrain_upsample_enabled

    pretrain_upsample_target_per_label = _parse_env_int(
        "CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL"
    )
    if pretrain_upsample_target_per_label is not None:
        if pretrain_upsample_target_per_label < 1:
            raise ValueError(
                "CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL must be at least 1."
            )
        cfg.pretrain_upsample_target_per_label = pretrain_upsample_target_per_label

    pretrain_upsample_perturbations = os.getenv(
        "CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS"
    )
    if (
        pretrain_upsample_perturbations is not None
        and pretrain_upsample_perturbations.strip() != ""
    ):
        cfg.pretrain_upsample_perturbations = [
            name.strip()
            for name in pretrain_upsample_perturbations.split(",")
            if name.strip()
        ]

    pretrain_upsample_perturbations_per_sample = _parse_env_int(
        "CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE"
    )
    if pretrain_upsample_perturbations_per_sample is not None:
        if pretrain_upsample_perturbations_per_sample < 1:
            raise ValueError(
                "CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE must be at least 1."
            )
        cfg.pretrain_upsample_perturbations_per_sample = (
            pretrain_upsample_perturbations_per_sample
        )

    pretrain_multicod_synthetic_ratio = _parse_env_float(
        "CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_RATIO"
    )
    if pretrain_multicod_synthetic_ratio is not None:
        if pretrain_multicod_synthetic_ratio < 0:
            raise ValueError(
                "CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_RATIO must be non-negative."
            )
        cfg.pretrain_multicod_synthetic_ratio = pretrain_multicod_synthetic_ratio

    pretrain_multicod_synthetic_text_separator = os.getenv(
        "CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_TEXT_SEPARATOR"
    )
    if pretrain_multicod_synthetic_text_separator is not None:
        if pretrain_multicod_synthetic_text_separator == "":
            raise ValueError(
                "CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_TEXT_SEPARATOR must not be empty."
            )
        cfg.pretrain_multicod_synthetic_text_separator = (
            pretrain_multicod_synthetic_text_separator
        )

    masterlist_inject_enabled = _parse_env_bool("CODLLM_MASTERLIST_INJECT_ENABLED")
    if masterlist_inject_enabled is not None:
        cfg.masterlist_inject_enabled = masterlist_inject_enabled

    masterlist_inject_target_per_label = _parse_env_int(
        "CODLLM_MASTERLIST_INJECT_TARGET_PER_LABEL"
    )
    if masterlist_inject_target_per_label is not None:
        if masterlist_inject_target_per_label < 1:
            raise ValueError(
                "CODLLM_MASTERLIST_INJECT_TARGET_PER_LABEL must be at least 1."
            )
        cfg.masterlist_inject_target_per_label = masterlist_inject_target_per_label

    masterlist_inject_perturbations = os.getenv(
        "CODLLM_MASTERLIST_INJECT_PERTURBATIONS"
    )
    if (
        masterlist_inject_perturbations is not None
        and masterlist_inject_perturbations.strip() != ""
    ):
        cfg.masterlist_inject_perturbations = [
            name.strip()
            for name in masterlist_inject_perturbations.split(",")
            if name.strip()
        ]

    masterlist_inject_perturbations_per_sample = _parse_env_int(
        "CODLLM_MASTERLIST_INJECT_PERTURBATIONS_PER_SAMPLE"
    )
    if masterlist_inject_perturbations_per_sample is not None:
        if masterlist_inject_perturbations_per_sample < 1:
            raise ValueError(
                "CODLLM_MASTERLIST_INJECT_PERTURBATIONS_PER_SAMPLE must be at least 1."
            )
        cfg.masterlist_inject_perturbations_per_sample = (
            masterlist_inject_perturbations_per_sample
        )

    inference_validate_registry = _parse_env_bool("CODLLM_INFERENCE_VALIDATE_REGISTRY")
    if inference_validate_registry is not None:
        cfg.inference_validate_registry = inference_validate_registry

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
    _apply_input_prefix_env(cfg)

    data_raw_dir = os.getenv("CODLLM_DATA_RAW_DIR")
    if data_raw_dir:
        cfg.data_raw_dir = data_raw_dir

    data_processed_dir = os.getenv("CODLLM_DATA_PROCESSED_DIR")
    if data_processed_dir:
        cfg.data_processed_dir = data_processed_dir

    balance_strategy = os.getenv("CODLLM_BALANCE_STRATEGY")
    if balance_strategy is not None and balance_strategy.strip() != "":
        normalized_balance = balance_strategy.strip().lower()
        allowed_balance = {"none", "floor"}
        if normalized_balance not in allowed_balance:
            allowed = ", ".join(sorted(allowed_balance))
            raise ValueError(f"CODLLM_BALANCE_STRATEGY must be one of: {allowed}.")
        cfg.balance_strategy = cast(BalanceStrategy, normalized_balance)

    balance_perturbations = os.getenv("CODLLM_BALANCE_PERTURBATIONS")
    if balance_perturbations is not None and balance_perturbations.strip() != "":
        cfg.balance_perturbations = [
            p.strip() for p in balance_perturbations.split(",") if p.strip()
        ]

    balance_perturbation_mean = _parse_env_float("CODLLM_BALANCE_PERTURBATION_MEAN")
    if balance_perturbation_mean is not None:
        if balance_perturbation_mean < 0:
            raise ValueError("CODLLM_BALANCE_PERTURBATION_MEAN must be non-negative.")
        cfg.balance_perturbation_mean = balance_perturbation_mean

    balance_perturbation_variance = _parse_env_float(
        "CODLLM_BALANCE_PERTURBATION_VARIANCE"
    )
    if balance_perturbation_variance is not None:
        if balance_perturbation_variance < 0:
            raise ValueError(
                "CODLLM_BALANCE_PERTURBATION_VARIANCE must be non-negative."
            )
        cfg.balance_perturbation_variance = balance_perturbation_variance

    base_perturbations = os.getenv("CODLLM_BASE_PERTURBATIONS")
    if base_perturbations is not None and base_perturbations.strip() != "":
        cfg.base_perturbations = [
            p.strip() for p in base_perturbations.split(",") if p.strip()
        ]

    base_perturbation_mean = _parse_env_float("CODLLM_BASE_PERTURBATION_MEAN")
    if base_perturbation_mean is not None:
        if base_perturbation_mean < 0:
            raise ValueError("CODLLM_BASE_PERTURBATION_MEAN must be non-negative.")
        cfg.base_perturbation_mean = base_perturbation_mean

    base_perturbation_variance = _parse_env_float("CODLLM_BASE_PERTURBATION_VARIANCE")
    if base_perturbation_variance is not None:
        if base_perturbation_variance < 0:
            raise ValueError("CODLLM_BASE_PERTURBATION_VARIANCE must be non-negative.")
        cfg.base_perturbation_variance = base_perturbation_variance

    balance_floor = _parse_env_int("CODLLM_BALANCE_FLOOR")
    if balance_floor is not None:
        if balance_floor < 0:
            raise ValueError("CODLLM_BALANCE_FLOOR must be non-negative.")
        cfg.balance_floor = balance_floor

    balance_floor_decay = _parse_env_float("CODLLM_BALANCE_FLOOR_DECAY")
    if balance_floor_decay is not None:
        if balance_floor_decay < 0 or balance_floor_decay > 1:
            raise ValueError("CODLLM_BALANCE_FLOOR_DECAY must be between 0 and 1.")
        cfg.balance_floor_decay = balance_floor_decay

    base_perturbation_rate = _parse_env_float("CODLLM_BASE_PERTURBATION_RATE")
    legacy_balance_base_perturbation_rate = (
        _parse_env_float("CODLLM_BALANCE_BASE_PERTURBATION_RATE")
        if base_perturbation_rate is None
        else None
    )
    if (
        base_perturbation_rate is None
        and legacy_balance_base_perturbation_rate is not None
    ):
        warnings.warn(
            (
                "CODLLM_BALANCE_BASE_PERTURBATION_RATE is deprecated; use "
                "CODLLM_BASE_PERTURBATION_RATE for whole-training-set perturbation."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        base_perturbation_rate = legacy_balance_base_perturbation_rate
        if base_perturbations is None and balance_perturbations is not None:
            warnings.warn(
                (
                    "CODLLM_BALANCE_PERTURBATIONS now controls only floor-upsampled "
                    "copies; set CODLLM_BASE_PERTURBATIONS for whole-training-set "
                    "perturbation. Reusing the balance perturbation list for this "
                    "legacy configuration."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            cfg.base_perturbations = list(cfg.balance_perturbations)
        if base_perturbation_mean is None and balance_perturbation_mean is not None:
            cfg.base_perturbation_mean = cfg.balance_perturbation_mean
        if (
            base_perturbation_variance is None
            and balance_perturbation_variance is not None
        ):
            cfg.base_perturbation_variance = cfg.balance_perturbation_variance
    if base_perturbation_rate is not None:
        if base_perturbation_rate < 0 or base_perturbation_rate > 1:
            raise ValueError("CODLLM_BASE_PERTURBATION_RATE must be between 0 and 1.")
        cfg.base_perturbation_rate = base_perturbation_rate

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

    wandb_run_config_mode = os.getenv("CODLLM_WANDB_RUN_CONFIG_MODE")
    if wandb_run_config_mode is not None and wandb_run_config_mode.strip() != "":
        normalized_run_config_mode = wandb_run_config_mode.strip().lower()
        allowed_run_config_modes = {"minimal", "standard", "full"}
        if normalized_run_config_mode not in allowed_run_config_modes:
            allowed = ", ".join(sorted(allowed_run_config_modes))
            raise ValueError(f"CODLLM_WANDB_RUN_CONFIG_MODE must be one of: {allowed}.")
        cfg.wandb.run_config_mode = cast(WandbRunConfigMode, normalized_run_config_mode)

    wandb_metric_mode = os.getenv("CODLLM_WANDB_METRIC_MODE")
    if wandb_metric_mode is not None and wandb_metric_mode.strip() != "":
        normalized_metric_mode = wandb_metric_mode.strip().lower()
        allowed_metric_modes = {"core", "standard", "all"}
        if normalized_metric_mode not in allowed_metric_modes:
            allowed = ", ".join(sorted(allowed_metric_modes))
            raise ValueError(f"CODLLM_WANDB_METRIC_MODE must be one of: {allowed}.")
        cfg.wandb.metric_mode = cast(WandbMetricMode, normalized_metric_mode)

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
