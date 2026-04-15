import sys
from types import ModuleType
from typing import Any

import pytest
from transformers import BitsAndBytesConfig

import codllm.models.loaders as model_registry
from codllm.config import Config


def _mock_api_keys_module(
    monkeypatch: pytest.MonkeyPatch,
    hugging_face: str,
) -> None:
    """Inject a temporary codllm.api_keys module for token resolution tests."""
    api_keys_module = ModuleType("codllm.api_keys")
    api_keys_module.hugging_face = hugging_face
    monkeypatch.setitem(sys.modules, "codllm.api_keys", api_keys_module)


def test_resolve_hf_token_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config token should win over environment and api_keys values."""
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "from_env")
    _mock_api_keys_module(monkeypatch, hugging_face="from_file")
    cfg = Config(hf_token="from_config")
    assert model_registry._resolve_hf_token(cfg) == "from_config"


def test_resolve_hf_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment token should be used when config token is unset."""
    monkeypatch.setenv("HF_TOKEN", "from_env")
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    _mock_api_keys_module(monkeypatch, hugging_face="")
    cfg = Config(hf_token=None)
    assert model_registry._resolve_hf_token(cfg) == "from_env"


def test_resolve_hf_token_from_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_keys token should be used when config and environment are unset."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    _mock_api_keys_module(monkeypatch, hugging_face="from_file")
    cfg = Config(hf_token=None)
    assert model_registry._resolve_hf_token(cfg) is None


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
        device=model_registry.torch.device("cuda"),
        load_in_8bit=True,
        torch_dtype="float16",
    )
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: True)
    kwargs = model_registry._build_model_kwargs(cfg, token="token")
    assert kwargs["token"] == "token"
    assert kwargs["device_map"] == "auto"
    assert kwargs["trust_remote_code"] is True
    assert kwargs["use_safetensors"] is False
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
    assert kwargs["use_safetensors"] is False
    assert "device_map" not in kwargs
    assert "quantization_config" not in kwargs


def test_build_model_kwargs_drops_auto_device_map_on_non_cuda_device() -> None:
    """Auto device_map should be disabled when config device is not CUDA."""
    cfg = Config(
        device_map="auto",
        device=model_registry.torch.device("cpu"),
        load_in_8bit=False,
    )
    kwargs = model_registry._build_model_kwargs(cfg, token=None)
    assert "device_map" not in kwargs


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


def test_apply_hf_runtime_env_disables_safetensors_conversion() -> None:
    """Runtime env should disable safetensors auto-conversion when configured."""
    cfg = Config(disable_safetensors_conversion=True)
    model_registry._apply_hf_runtime_env(cfg)
    assert model_registry.os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_flan_t5_small_uses_default_seq2seq_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flan-t5-small should load through the default seq2seq path."""
    cfg = Config(hf_model="google/flan-t5-small", use_safetensors=False)
    captured: dict[str, bool] = {}

    def fake_default_seq2seq(loader_cfg: Config) -> tuple[Any, Any]:
        captured["use_safetensors"] = loader_cfg.use_safetensors
        return object(), object()

    monkeypatch.delitem(model_registry.MODEL_REGISTRY, cfg.hf_model, raising=False)
    monkeypatch.setattr(model_registry, "load_default_seq2seq", fake_default_seq2seq)
    model_registry.load_base_model(cfg)

    assert captured["use_safetensors"] is False


def test_load_base_model_uses_sequence_classification_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequence-classification task should load the classifier model path."""
    cfg = Config(
        hf_model="google/flan-t5-small",
        model_task="sequence_classification",
    )
    captured: dict[str, Any] = {}

    def fake_classifier_loader(
        loader_cfg: Config,
        label2id: dict[str, int] | None = None,
        id2label: dict[int, str] | None = None,
    ) -> tuple[Any, Any]:
        captured["cfg"] = loader_cfg
        captured["label2id"] = label2id
        captured["id2label"] = id2label
        return object(), object()

    monkeypatch.setattr(
        model_registry,
        "load_default_sequence_classifier",
        fake_classifier_loader,
    )

    model_registry.load_base_model(
        cfg=cfg,
        label2id={"A00": 0},
        id2label={0: "A00"},
    )

    assert captured["cfg"] is cfg
    assert captured["label2id"] == {"A00": 0}
    assert captured["id2label"] == {0: "A00"}
