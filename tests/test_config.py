import pytest

from codllm.config import Config, config_from_env


ENV_KEYS = [
    "CODLLM_SEED",
    "CODLLM_DATA_SEED",
    "CODLLM_DATALOADER_NUM_WORKERS",
    "CODLLM_WARMUP_STEPS",
    "CODLLM_NUM_TRAIN_EPOCHS",
    "CODLLM_MAX_GRAD_NORM",
    "CODLLM_WEIGHT_DECAY",
    "CODLLM_MAX_LABEL_COUNT",
    "CODLLM_MAX_TARGET_LENGTH",
    "CODLLM_LABEL_CODE_LENGTH",
    "CODLLM_MAX_TARGET_LENGTH_BUFFER",
    "CODLLM_LABEL_SEPARATOR",
    "CODLLM_TRAINING_INPUT",
    "CODLLM_DETERMINISTIC_ALGORITHMS",
    "CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY",
    "CODLLM_CUDNN_DETERMINISTIC",
    "CODLLM_CUDNN_BENCHMARK",
    "CODLLM_LOAD_IN_8BIT",
    "CODLLM_VERBOSE",
    "CODLLM_TORCH_DTYPE",
    "CODLLM_LR",
    "CODLLM_DATASET_SIZE",
    "CODLLM_TRAIN_SIZE",
    "CODLLM_VAL_SIZE",
    "CODLLM_TEST_SIZE",
    "CODLLM_OUTPUT_DIR",
    "CODLLM_DATA_RAW_DIR",
    "CODLLM_DATA_PROCESSED_DIR",
    "CODLLM_WANDB_LOG_MODEL",
]


def _clear_relevant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear config override env vars for an isolated test."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_config_from_env_applies_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables should override reproducibility and path settings."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_SEED", "101")
    monkeypatch.setenv("CODLLM_DATA_SEED", "202")
    monkeypatch.setenv("CODLLM_DATALOADER_NUM_WORKERS", "3")
    monkeypatch.setenv("CODLLM_WARMUP_STEPS", "500")
    monkeypatch.setenv("CODLLM_NUM_TRAIN_EPOCHS", "6")
    monkeypatch.setenv("CODLLM_MAX_GRAD_NORM", "0.25")
    monkeypatch.setenv("CODLLM_WEIGHT_DECAY", "0.03")
    monkeypatch.setenv("CODLLM_MAX_LABEL_COUNT", "2")
    monkeypatch.setenv("CODLLM_MAX_TARGET_LENGTH", "18")
    monkeypatch.setenv("CODLLM_LABEL_CODE_LENGTH", "7")
    monkeypatch.setenv("CODLLM_MAX_TARGET_LENGTH_BUFFER", "6")
    monkeypatch.setenv("CODLLM_LABEL_SEPARATOR", ",")
    monkeypatch.setenv("CODLLM_TRAINING_INPUT", "cod,age")
    monkeypatch.setenv("CODLLM_DETERMINISTIC_ALGORITHMS", "true")
    monkeypatch.setenv("CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY", "false")
    monkeypatch.setenv("CODLLM_CUDNN_DETERMINISTIC", "false")
    monkeypatch.setenv("CODLLM_CUDNN_BENCHMARK", "true")
    monkeypatch.setenv("CODLLM_LOAD_IN_8BIT", "true")
    monkeypatch.setenv("CODLLM_VERBOSE", "true")
    monkeypatch.setenv("CODLLM_TORCH_DTYPE", "float32")
    monkeypatch.setenv("CODLLM_LR", "5e-5")
    monkeypatch.setenv("CODLLM_DATASET_SIZE", "0.75")
    monkeypatch.setenv("CODLLM_TRAIN_SIZE", "0.7")
    monkeypatch.setenv("CODLLM_VAL_SIZE", "0.2")
    monkeypatch.setenv("CODLLM_TEST_SIZE", "0.1")
    monkeypatch.setenv("CODLLM_OUTPUT_DIR", "/tmp/output")
    monkeypatch.setenv("CODLLM_DATA_RAW_DIR", "/tmp/raw")
    monkeypatch.setenv("CODLLM_DATA_PROCESSED_DIR", "/tmp/processed")
    monkeypatch.setenv("CODLLM_WANDB_LOG_MODEL", "checkpoint")

    base = Config(seed=42, data_seed=None, output_dir="./runs", load_in_8bit=False)
    cfg = config_from_env(base)

    assert cfg.seed == 101
    assert cfg.data_seed == 202
    assert cfg.dataloader_num_workers == 3
    assert cfg.warmup_steps == 500
    assert cfg.num_train_epochs == 6
    assert cfg.max_grad_norm == 0.25
    assert cfg.weight_decay == 0.03
    assert cfg.max_label_count == 2
    assert cfg.max_target_length == 18
    assert cfg.label_code_length == 7
    assert cfg.max_target_length_buffer == 6
    assert cfg.label_separator == ","
    assert cfg.training_input == ["cod", "age"]
    assert cfg.deterministic_algorithms is True
    assert cfg.deterministic_algorithms_warn_only is False
    assert cfg.cudnn_deterministic is False
    assert cfg.cudnn_benchmark is True
    assert cfg.load_in_8bit is True
    assert cfg.verbose is True
    assert cfg.torch_dtype == "float32"
    assert cfg.lr == 5e-5
    assert cfg.dataset_size == 0.75
    assert cfg.train_size == 0.7
    assert cfg.val_size == 0.2
    assert cfg.test_size == 0.1
    assert cfg.output_dir == "/tmp/output"
    assert cfg.data_raw_dir == "/tmp/raw"
    assert cfg.data_processed_dir == "/tmp/processed"
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


def test_config_from_env_rejects_invalid_training_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """training_input override should reject unsupported values."""
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("CODLLM_TRAINING_INPUT", "cod,city")
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
