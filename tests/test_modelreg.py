from typing import Any

import pytest
from transformers import BitsAndBytesConfig

import src.model_registry as model_registry
from config import Config


def test_resolve_hf_token_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config token should win over environment and api_keys values."""
    import api_keys

    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "from_env")
    monkeypatch.setattr(api_keys, "hugging_face", "from_file")
    cfg = Config(hf_token="from_config")
    assert model_registry._resolve_hf_token(cfg) == "from_config"


def test_resolve_hf_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment token should be used when config token is unset."""
    import api_keys

    monkeypatch.setenv("HF_TOKEN", "from_env")
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(api_keys, "hugging_face", "")
    cfg = Config(hf_token=None)
    assert model_registry._resolve_hf_token(cfg) == "from_env"


def test_resolve_hf_token_from_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_keys token should be used when config and environment are unset."""
    import api_keys

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(api_keys, "hugging_face", "from_file")
    cfg = Config(hf_token=None)
    assert model_registry._resolve_hf_token(cfg) == "from_file"


def test_resolve_torch_dtype_known_values() -> None:
    """Known dtype names should map to accepted transformers values."""
    assert model_registry._resolve_torch_dtype("float16") is not None
    assert model_registry._resolve_torch_dtype("float32") is not None
    assert model_registry._resolve_torch_dtype("bfloat16") is not None
    assert model_registry._resolve_torch_dtype("auto") == "auto"
    assert model_registry._resolve_torch_dtype(None) is None


def test_resolve_torch_dtype_rejects_unknown_value() -> None:
    """Unknown dtype names should raise a clear ValueError."""
    with pytest.raises(ValueError):
        model_registry._resolve_torch_dtype("fp8")


def test_build_model_kwargs_with_cuda_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model kwargs should include quantization only when CUDA is available."""
    cfg = Config(
        hf_model="google/flan-ul2",
        hf_token="token",
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype="float16",
    )
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: True)
    kwargs = model_registry._build_model_kwargs(cfg, token="token")
    assert kwargs["token"] == "token"
    assert kwargs["device_map"] == "auto"
    assert kwargs["trust_remote_code"] is True
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)


def test_build_model_kwargs_without_cuda_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model kwargs should not include quantization config when CUDA is unavailable."""
    cfg = Config(
        hf_model="google/flan-ul2",
        hf_token="token",
        trust_remote_code=False,
        device_map=None,
        load_in_8bit=True,
        torch_dtype="float32",
    )
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: False)
    kwargs = model_registry._build_model_kwargs(cfg, token="token")
    assert kwargs["token"] == "token"
    assert kwargs["trust_remote_code"] is False
    assert "device_map" not in kwargs
    assert "quantization_config" not in kwargs


def test_load_base_model_uses_registry_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registered loaders should override the default loader path."""
    cfg = Config(hf_model="unit-test/custom")

    class DummyModel:
        """Dummy model for loader tests."""

    class DummyTokenizer:
        """Dummy tokenizer for loader tests."""

    def fake_loader(loader_cfg: Config) -> tuple[Any, Any]:
        assert loader_cfg is cfg
        return DummyModel(), DummyTokenizer()

    monkeypatch.setitem(model_registry.MODEL_REGISTRY, cfg.hf_model, fake_loader)
    model, tokenizer = model_registry.load_base_model(cfg)
    assert isinstance(model, DummyModel)
    assert isinstance(tokenizer, DummyTokenizer)
