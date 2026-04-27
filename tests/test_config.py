import pytest

from codllm.config import Config, config_from_env


ENV_KEYS = [
    "CODLLM_HF_MODEL",
    "CODLLM_HF_TOKEN",
    "CODLLM_TRUST_REMOTE_CODE",
    "CODLLM_SEED",
    "CODLLM_DATA_SEED",
    "CODLLM_DATALOADER_NUM_WORKERS",
    "CODLLM_DATALOADER_PIN_MEMORY",
    "CODLLM_DATALOADER_PERSISTENT_WORKERS",
    "CODLLM_DATALOADER_PREFETCH_FACTOR",
    "CODLLM_MAX_SOURCE_LENGTH",
    "CODLLM_WARMUP_STEPS",
    "CODLLM_NUM_TRAIN_EPOCHS",
    "CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE",
    "CODLLM_PER_DEVICE_EVAL_BATCH_SIZE",
    "CODLLM_GRADIENT_ACCUMULATION_STEPS",
    "CODLLM_LOGGING_STEPS",
    "CODLLM_EVAL_STEPS",
    "CODLLM_SAVE_STEPS",
    "CODLLM_EVAL_STRATEGY",
    "CODLLM_SAVE_STRATEGY",
    "CODLLM_SAVE_STRATEGY_BEST_METRIC",
    "CODLLM_MODEL_TASK",
    "CODLLM_LR_SCHEDULER_TYPE",
    "CODLLM_MAX_GRAD_NORM",
    "CODLLM_WEIGHT_DECAY",
    "CODLLM_MAX_LABEL_COUNT",
    "CODLLM_INPUT_PREFIX_COD",
    "CODLLM_INPUT_PREFIX_AGE",
    "CODLLM_INPUT_PREFIX_SEX",
    "CODLLM_MULTICOD_SHUFFLE_LABELS",
    "CODLLM_MULTICOD_SYNTHETIC_RATIO",
    "CODLLM_MULTICOD_SYNTHETIC_SOURCE_SCOPE",
    "CODLLM_MULTICOD_SYNTHETIC_TEXT_SEPARATOR",
    "CODLLM_MAX_TARGET_LENGTH",
    "CODLLM_LABEL_CODE_LENGTH",
    "CODLLM_MAX_TARGET_LENGTH_BUFFER",
    "CODLLM_LABEL_SEPARATOR",
    "CODLLM_TEXT_FIELD_SEPARATOR",
    "CODLLM_TRAINING_INPUT",
    "CODLLM_BALANCE_STRATEGY",
    "CODLLM_BALANCE_TARGET_QUANTILE",
    "CODLLM_BALANCE_PERTURBATIONS",
    "CODLLM_BALANCE_PERTURBATIONS_PER_SAMPLE",
    "CODLLM_BALANCE_UPSAMPLE_LABELS",
    "CODLLM_BALANCE_UPSAMPLE_INVERSE_POWER",
    "CODLLM_BALANCE_UPSAMPLE_BUDGET_RATIO",
    "CODLLM_BALANCE_BASE_PERTURBATION_RATE",
    "CODLLM_DETERMINISTIC_ALGORITHMS",
    "CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY",
    "CODLLM_CUDNN_DETERMINISTIC",
    "CODLLM_CUDNN_BENCHMARK",
    "CODLLM_LOAD_IN_8BIT",
    "CODLLM_USE_SAFETENSORS",
    "CODLLM_DISABLE_SAFETENSORS_CONVERSION",
    "CODLLM_VERBOSE",
    "CODLLM_TORCH_DTYPE",
    "CODLLM_LR",
    "CODLLM_DATASET_SIZE",
    "CODLLM_TRAIN_SIZE",
    "CODLLM_VAL_SIZE",
    "CODLLM_TEST_SIZE",
    "CODLLM_PRETRAIN_ENABLED",
    "CODLLM_PRETRAIN_MASTERLIST_PATH",
    "CODLLM_PRETRAIN_MASTERLIST_SHEET_NAME",
    "CODLLM_PRETRAIN_TRANSFER_SHEET_NAME",
    "CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS",
    "CODLLM_PRETRAIN_LEARNING_RATE",
    "CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS",
    "CODLLM_PRETRAIN_LR_SCHEDULER_TYPE",
    "CODLLM_PRETRAIN_UPSAMPLE_ENABLED",
    "CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL",
    "CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS",
    "CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE",
    "CODLLM_LABEL_HARMONIZATION_ENABLED",
    "CODLLM_INFERENCE_VALIDATE_REGISTRY",
    "CODLLM_OUTPUT_DIR",
    "CODLLM_DATA_RAW_DIR",
    "CODLLM_DATA_PROCESSED_DIR",
    "CODLLM_DATASET_TEXT_COLUMN",
    "CODLLM_DATASET_LABEL_COLUMN",
    "CODLLM_DEVICE",
    "CODLLM_DEVICE_MAP",
    "CODLLM_WANDB_ENABLED",
    "CODLLM_WANDB_MODE",
    "CODLLM_WANDB_PROJECT",
    "CODLLM_WANDB_ENTITY",
    "CODLLM_WANDB_RUN_NAME",
    "CODLLM_WANDB_LOG_MODEL",
]


def _clear_relevant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear config override env vars for an isolated test."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_config_defaults_use_stable_seq2seq_training_baseline() -> None:
    """Defaults should keep simplified run specs aligned with the stable seq2seq baseline."""
    cfg = Config()

    assert cfg.model_task == "seq2seq"
    assert cfg.training_input == ["cod", "age", "sex"]
    assert cfg.input_field_prefixes == {
        "cod": "cod: ",
        "age": "age: ",
        "sex": "sex: ",
    }
    assert cfg.max_grad_norm == 0.5
    assert cfg.warmup_steps == 1000
    assert cfg.balance_strategy == "none"


def test_config_from_env_applies_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables should override reproducibility and path settings."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_HF_MODEL", "google/flan-t5-base")
    monkeypatch.setenv("CODLLM_HF_TOKEN", "test-token")
    monkeypatch.setenv("CODLLM_TRUST_REMOTE_CODE", "true")
    monkeypatch.setenv("CODLLM_SEED", "101")
    monkeypatch.setenv("CODLLM_DATA_SEED", "202")
    monkeypatch.setenv("CODLLM_DATALOADER_NUM_WORKERS", "3")
    monkeypatch.setenv("CODLLM_DATALOADER_PIN_MEMORY", "false")
    monkeypatch.setenv("CODLLM_DATALOADER_PERSISTENT_WORKERS", "true")
    monkeypatch.setenv("CODLLM_DATALOADER_PREFETCH_FACTOR", "4")
    monkeypatch.setenv("CODLLM_MAX_SOURCE_LENGTH", "300")
    monkeypatch.setenv("CODLLM_WARMUP_STEPS", "500")
    monkeypatch.setenv("CODLLM_NUM_TRAIN_EPOCHS", "6")
    monkeypatch.setenv("CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE", "6")
    monkeypatch.setenv("CODLLM_PER_DEVICE_EVAL_BATCH_SIZE", "5")
    monkeypatch.setenv("CODLLM_GRADIENT_ACCUMULATION_STEPS", "3")
    monkeypatch.setenv("CODLLM_LOGGING_STEPS", "12")
    monkeypatch.setenv("CODLLM_EVAL_STEPS", "50")
    monkeypatch.setenv("CODLLM_SAVE_STEPS", "60")
    monkeypatch.setenv("CODLLM_EVAL_STRATEGY", "steps")
    monkeypatch.setenv("CODLLM_SAVE_STRATEGY", "best")
    monkeypatch.setenv("CODLLM_SAVE_STRATEGY_BEST_METRIC", "macro_f1")
    monkeypatch.setenv("CODLLM_MODEL_TASK", "sequence_classification")
    monkeypatch.setenv("CODLLM_LR_SCHEDULER_TYPE", "cosine")
    monkeypatch.setenv("CODLLM_MAX_GRAD_NORM", "0.25")
    monkeypatch.setenv("CODLLM_WEIGHT_DECAY", "0.03")
    monkeypatch.setenv("CODLLM_MAX_LABEL_COUNT", "2")
    monkeypatch.setenv("CODLLM_INPUT_PREFIX_COD", "cause=")
    monkeypatch.setenv("CODLLM_INPUT_PREFIX_AGE", "years=")
    monkeypatch.setenv("CODLLM_INPUT_PREFIX_SEX", "gender=")
    monkeypatch.setenv("CODLLM_MULTICOD_SHUFFLE_LABELS", "false")
    monkeypatch.setenv("CODLLM_MULTICOD_SYNTHETIC_RATIO", "0.25")
    monkeypatch.setenv("CODLLM_MULTICOD_SYNTHETIC_SOURCE_SCOPE", "any_source")
    monkeypatch.setenv("CODLLM_MULTICOD_SYNTHETIC_TEXT_SEPARATOR", " + ")
    monkeypatch.setenv("CODLLM_MAX_TARGET_LENGTH", "18")
    monkeypatch.setenv("CODLLM_LABEL_CODE_LENGTH", "7")
    monkeypatch.setenv("CODLLM_MAX_TARGET_LENGTH_BUFFER", "6")
    monkeypatch.setenv("CODLLM_LABEL_SEPARATOR", ",")
    monkeypatch.setenv("CODLLM_TEXT_FIELD_SEPARATOR", " || ")
    monkeypatch.setenv("CODLLM_TRAINING_INPUT", "cod,age")
    monkeypatch.setenv("CODLLM_BALANCE_STRATEGY", "upsample")
    monkeypatch.setenv("CODLLM_BALANCE_TARGET_QUANTILE", "0.6")
    monkeypatch.setenv("CODLLM_BALANCE_PERTURBATIONS", "delete_random_char")
    monkeypatch.setenv("CODLLM_BALANCE_PERTURBATIONS_PER_SAMPLE", "2")
    monkeypatch.setenv("CODLLM_BALANCE_UPSAMPLE_LABELS", "A00,A01")
    monkeypatch.setenv("CODLLM_BALANCE_UPSAMPLE_INVERSE_POWER", "0.6")
    monkeypatch.setenv("CODLLM_BALANCE_UPSAMPLE_BUDGET_RATIO", "0.25")
    monkeypatch.setenv("CODLLM_BALANCE_BASE_PERTURBATION_RATE", "0.5")
    monkeypatch.setenv("CODLLM_DETERMINISTIC_ALGORITHMS", "true")
    monkeypatch.setenv("CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY", "false")
    monkeypatch.setenv("CODLLM_CUDNN_DETERMINISTIC", "false")
    monkeypatch.setenv("CODLLM_CUDNN_BENCHMARK", "true")
    monkeypatch.setenv("CODLLM_LOAD_IN_8BIT", "true")
    monkeypatch.setenv("CODLLM_USE_SAFETENSORS", "true")
    monkeypatch.setenv("CODLLM_DISABLE_SAFETENSORS_CONVERSION", "false")
    monkeypatch.setenv("CODLLM_VERBOSE", "true")
    monkeypatch.setenv("CODLLM_TORCH_DTYPE", "float32")
    monkeypatch.setenv("CODLLM_LR", "5e-5")
    monkeypatch.setenv("CODLLM_DATASET_SIZE", "0.75")
    monkeypatch.setenv("CODLLM_TRAIN_SIZE", "0.7")
    monkeypatch.setenv("CODLLM_VAL_SIZE", "0.2")
    monkeypatch.setenv("CODLLM_TEST_SIZE", "0.1")
    monkeypatch.setenv("CODLLM_PRETRAIN_ENABLED", "true")
    monkeypatch.setenv(
        "CODLLM_PRETRAIN_MASTERLIST_PATH", "data/raw/ICD10h_Masterlist_2024.xlsx"
    )
    monkeypatch.setenv("CODLLM_PRETRAIN_MASTERLIST_SHEET_NAME", "Masterlist")
    monkeypatch.setenv("CODLLM_PRETRAIN_TRANSFER_SHEET_NAME", "2020to2024transfer")
    monkeypatch.setenv("CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS", "2")
    monkeypatch.setenv("CODLLM_PRETRAIN_LEARNING_RATE", "8e-6")
    monkeypatch.setenv("CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS", "10")
    monkeypatch.setenv("CODLLM_PRETRAIN_LR_SCHEDULER_TYPE", "linear")
    monkeypatch.setenv("CODLLM_PRETRAIN_UPSAMPLE_ENABLED", "true")
    monkeypatch.setenv("CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL", "10")
    monkeypatch.setenv(
        "CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS",
        "delete_random_char,qwerty_misspell",
    )
    monkeypatch.setenv("CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE", "2")
    monkeypatch.setenv("CODLLM_LABEL_HARMONIZATION_ENABLED", "true")
    monkeypatch.setenv("CODLLM_INFERENCE_VALIDATE_REGISTRY", "true")
    monkeypatch.setenv("CODLLM_OUTPUT_DIR", "/tmp/output")
    monkeypatch.setenv("CODLLM_DATA_RAW_DIR", "/tmp/raw")
    monkeypatch.setenv("CODLLM_DATA_PROCESSED_DIR", "/tmp/processed")
    monkeypatch.setenv("CODLLM_DATASET_TEXT_COLUMN", "prompt")
    monkeypatch.setenv("CODLLM_DATASET_LABEL_COLUMN", "target")
    monkeypatch.setenv("CODLLM_DEVICE", "cpu")
    monkeypatch.setenv("CODLLM_DEVICE_MAP", "none")
    monkeypatch.setenv("CODLLM_WANDB_ENABLED", "false")
    monkeypatch.setenv("CODLLM_WANDB_MODE", "offline")
    monkeypatch.setenv("CODLLM_WANDB_PROJECT", "codllm-tests")
    monkeypatch.setenv("CODLLM_WANDB_ENTITY", "unit-tests")
    monkeypatch.setenv("CODLLM_WANDB_RUN_NAME", "run-123")
    monkeypatch.setenv("CODLLM_WANDB_LOG_MODEL", "checkpoint")

    base = Config(seed=42, data_seed=None, output_dir="./runs", load_in_8bit=False)
    cfg = config_from_env(base)

    assert cfg.hf_model == "google/flan-t5-base"
    assert cfg.hf_token == "test-token"
    assert cfg.trust_remote_code is True
    assert cfg.seed == 101
    assert cfg.data_seed == 202
    assert cfg.dataloader_num_workers == 3
    assert cfg.dataloader_pin_memory is False
    assert cfg.dataloader_persistent_workers is True
    assert cfg.dataloader_prefetch_factor == 4
    assert cfg.max_source_length == 300
    assert cfg.warmup_steps == 500
    assert cfg.num_train_epochs == 6
    assert cfg.per_device_train_batch_size == 6
    assert cfg.per_device_eval_batch_size == 5
    assert cfg.gradient_accumulation_steps == 3
    assert cfg.logging_steps == 12
    assert cfg.eval_steps == 50
    assert cfg.save_steps == 60
    assert cfg.eval_strategy == "steps"
    assert cfg.save_strategy == "best"
    assert cfg.save_strategy_best_metric == "macro_f1"
    assert cfg.model_task == "sequence_classification"
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.max_grad_norm == 0.25
    assert cfg.weight_decay == 0.03
    assert cfg.max_label_count == 2
    assert cfg.input_field_prefixes == {
        "cod": "cause=",
        "age": "years=",
        "sex": "gender=",
    }
    assert cfg.multicod_shuffle_labels is False
    assert cfg.multicod_synthetic_ratio == 0.25
    assert cfg.multicod_synthetic_source_scope == "any_source"
    assert cfg.multicod_synthetic_text_separator == " + "
    assert cfg.max_target_length == 18
    assert cfg.label_code_length == 7
    assert cfg.max_target_length_buffer == 6
    assert cfg.label_separator == ","
    assert cfg.text_field_separator == " || "
    assert cfg.training_input == ["cod", "age"]
    assert cfg.balance_strategy == "upsample"
    assert cfg.balance_target_quantile == 0.6
    assert cfg.balance_perturbations == ["delete_random_char"]
    assert cfg.balance_perturbations_per_sample == 2
    assert cfg.balance_upsample_labels == ["A00", "A01"]
    assert cfg.balance_upsample_inverse_power == 0.6
    assert cfg.balance_upsample_budget_ratio == 0.25
    assert cfg.balance_base_perturbation_rate == 0.5
    assert cfg.deterministic_algorithms is True
    assert cfg.deterministic_algorithms_warn_only is False
    assert cfg.cudnn_deterministic is False
    assert cfg.cudnn_benchmark is True
    assert cfg.load_in_8bit is True
    assert cfg.use_safetensors is True
    assert cfg.disable_safetensors_conversion is False
    assert cfg.verbose is True
    assert cfg.torch_dtype == "float32"
    assert cfg.lr == 5e-5
    assert cfg.dataset_size == 0.75
    assert cfg.train_size == 0.7
    assert cfg.val_size == 0.2
    assert cfg.test_size == 0.1
    assert cfg.pretrain_enabled is True
    assert cfg.pretrain_masterlist_path == "data/raw/ICD10h_Masterlist_2024.xlsx"
    assert cfg.pretrain_masterlist_sheet_name == "Masterlist"
    assert cfg.pretrain_transfer_sheet_name == "2020to2024transfer"
    assert cfg.pretrain_num_train_epochs == 2
    assert cfg.pretrain_learning_rate == 8e-6
    assert cfg.pretrain_eval_every_n_epochs == 10
    assert cfg.pretrain_lr_scheduler_type == "linear"
    assert cfg.pretrain_upsample_enabled is True
    assert cfg.pretrain_upsample_target_per_label == 10
    assert cfg.pretrain_upsample_perturbations == [
        "delete_random_char",
        "qwerty_misspell",
    ]
    assert cfg.pretrain_upsample_perturbations_per_sample == 2
    assert cfg.label_harmonization_enabled is True
    assert cfg.inference_validate_registry is True
    assert cfg.output_dir == "/tmp/output"
    assert cfg.data_raw_dir == "/tmp/raw"
    assert cfg.data_processed_dir == "/tmp/processed"
    assert cfg.dataset_text_column == "prompt"
    assert cfg.dataset_label_column == "target"
    assert cfg.device.type == "cpu"
    assert cfg.device_map is None
    assert cfg.wandb.enabled is False
    assert cfg.wandb.mode == "offline"
    assert cfg.wandb.project == "codllm-tests"
    assert cfg.wandb.entity == "unit-tests"
    assert cfg.wandb.run_name == "run-123"
    assert cfg.wandb.log_model == "checkpoint"
    assert base.seed == 42
    assert base.output_dir == "./runs"


def test_config_from_env_rejects_invalid_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid boolean values should fail fast."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_DETERMINISTIC_ALGORITHMS", "sometimes")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_negative_dataloader_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataloader worker count should be non-negative."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_DATALOADER_NUM_WORKERS", "-1")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_dataloader_prefetch_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataloader prefetch factor should be at least one."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_DATALOADER_PREFETCH_FACTOR", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_negative_warmup_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup steps should be non-negative."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_WARMUP_STEPS", "-1")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_non_positive_num_train_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train epochs should be at least one."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_NUM_TRAIN_EPOCHS", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_non_positive_train_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-device train batch size should be at least one."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_non_positive_learning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Learning rate should be positive."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_LR", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_torch_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch dtype override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_TORCH_DTYPE", "fp8")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_eval_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eval strategy override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_EVAL_STRATEGY", "batch")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_save_strategy_best_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-save metric override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_SAVE_STRATEGY_BEST_METRIC", "bleu")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_DEVICE", "tpu")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_honors_verbose_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verbose override should disable terminal config dumps when false."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_VERBOSE", "false")
    cfg = config_from_env(Config(verbose=True))
    assert cfg.verbose is False


def test_config_from_env_rejects_negative_max_grad_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max grad norm should be non-negative."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_MAX_GRAD_NORM", "-0.1")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_negative_weight_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weight decay should be non-negative."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_WEIGHT_DECAY", "-0.01")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_wandb_log_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W&B log-model override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_WANDB_LOG_MODEL", "always")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_wandb_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W&B mode override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_WANDB_MODE", "local")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_training_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """training_input override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_TRAINING_INPUT", "cod,city")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_empty_input_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input prefix overrides should not allow empty strings."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_INPUT_PREFIX_COD", "")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_negative_multicod_synthetic_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic multi-COD ratio should be non-negative."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_MULTICOD_SYNTHETIC_RATIO", "-0.1")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_multicod_synthetic_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic multi-COD source scope should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_MULTICOD_SYNTHETIC_SOURCE_SCOPE", "cross_period")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_balance_base_perturbation_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base perturbation rate should stay inside [0, 1]."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_BALANCE_BASE_PERTURBATION_RATE", "1.5")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_upsample_inverse_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inverse power should stay inside (0, 1]."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_BALANCE_UPSAMPLE_INVERSE_POWER", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_upsample_budget_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upsample budget ratio should stay inside [0, 1]."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_BALANCE_UPSAMPLE_BUDGET_RATIO", "1.2")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_pretrain_num_train_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretraining epochs must be at least one."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_non_positive_pretrain_learning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretraining learning rate must be positive when provided."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_PRETRAIN_LEARNING_RATE", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_pretrain_eval_every_n_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretraining eval interval must be at least one epoch."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_pretrain_upsample_target_per_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretraining upsample target must be at least one."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_pretrain_perturbations_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretraining perturbations per sample must be at least one."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_lr_scheduler_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LR scheduler type should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_LR_SCHEDULER_TYPE", "invalid")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_model_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model task should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_MODEL_TASK", "invalid_task")
    with pytest.raises(ValueError):
        config_from_env()


def test_config_from_env_rejects_invalid_pretrain_lr_scheduler_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretraining LR scheduler type should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_PRETRAIN_LR_SCHEDULER_TYPE", "invalid")
    with pytest.raises(ValueError):
        config_from_env()


def test_resolved_max_target_length_uses_code_length_and_label_count() -> None:
    """Target max length should expand with multiple labels."""
    cfg = Config(
        max_target_length=16,
        max_label_count=2,
        label_code_length=7,
        label_separator=" | ",
        max_target_length_buffer=4,
    )
    assert cfg.resolved_max_target_length() == 21


def test_resolved_max_target_length_respects_manual_ceiling() -> None:
    """Configured max_target_length should remain when already larger."""
    cfg = Config(
        max_target_length=64,
        max_label_count=2,
        label_code_length=7,
        label_separator=" | ",
        max_target_length_buffer=4,
    )
    assert cfg.resolved_max_target_length() == 64


def test_resolved_data_seed_falls_back_to_seed() -> None:
    """Data seed helper should fall back to the global seed."""
    cfg = Config(seed=123, data_seed=None)
    assert cfg.resolved_data_seed() == 123


def test_resolved_data_seed_uses_explicit_value() -> None:
    """Data seed helper should use explicit data_seed when configured."""
    cfg = Config(seed=123, data_seed=456)
    assert cfg.resolved_data_seed() == 456


def test_default_data_sources_include_copenhagen_dataset() -> None:
    """Default source list should include the Danish Copenhagen dataset."""
    cfg = Config()
    by_source_id = {source.source_id: source for source in cfg.data_sources}
    assert "copenhagen_may2025" in by_source_id
    copenhagen = by_source_id["copenhagen_may2025"]
    assert copenhagen.path == "Copenhagen_burials_all_May2025.csv"
    assert copenhagen.mapping_id == "copenhagen"
