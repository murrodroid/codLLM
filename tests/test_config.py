import pytest

from codllm.config import Config, config_from_env


ENV_KEYS = [
    "CODLLM_SEED",
    "CODLLM_DATA_SEED",
    "CODLLM_DATALOADER_NUM_WORKERS",
    "CODLLM_DETERMINISTIC_ALGORITHMS",
    "CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY",
    "CODLLM_CUDNN_DETERMINISTIC",
    "CODLLM_CUDNN_BENCHMARK",
    "CODLLM_DATASET_SIZE",
    "CODLLM_TRAIN_SIZE",
    "CODLLM_VAL_SIZE",
    "CODLLM_TEST_SIZE",
    "CODLLM_OUTPUT_DIR",
    "CODLLM_DATA_RAW_DIR",
    "CODLLM_DATA_PROCESSED_DIR",
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
    monkeypatch.setenv("CODLLM_DETERMINISTIC_ALGORITHMS", "true")
    monkeypatch.setenv("CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY", "false")
    monkeypatch.setenv("CODLLM_CUDNN_DETERMINISTIC", "false")
    monkeypatch.setenv("CODLLM_CUDNN_BENCHMARK", "true")
    monkeypatch.setenv("CODLLM_DATASET_SIZE", "0.75")
    monkeypatch.setenv("CODLLM_TRAIN_SIZE", "0.7")
    monkeypatch.setenv("CODLLM_VAL_SIZE", "0.2")
    monkeypatch.setenv("CODLLM_TEST_SIZE", "0.1")
    monkeypatch.setenv("CODLLM_OUTPUT_DIR", "/tmp/output")
    monkeypatch.setenv("CODLLM_DATA_RAW_DIR", "/tmp/raw")
    monkeypatch.setenv("CODLLM_DATA_PROCESSED_DIR", "/tmp/processed")

    base = Config(seed=42, data_seed=None, output_dir="./runs")
    cfg = config_from_env(base)

    assert cfg.seed == 101
    assert cfg.data_seed == 202
    assert cfg.dataloader_num_workers == 3
    assert cfg.deterministic_algorithms is True
    assert cfg.deterministic_algorithms_warn_only is False
    assert cfg.cudnn_deterministic is False
    assert cfg.cudnn_benchmark is True
    assert cfg.dataset_size == 0.75
    assert cfg.train_size == 0.7
    assert cfg.val_size == 0.2
    assert cfg.test_size == 0.1
    assert cfg.output_dir == "/tmp/output"
    assert cfg.data_raw_dir == "/tmp/raw"
    assert cfg.data_processed_dir == "/tmp/processed"
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
