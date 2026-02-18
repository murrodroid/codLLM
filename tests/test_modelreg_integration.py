import os

import pytest

from config import Config
from src.model_registry import load_base_model


RUN_HF_SMOKE = os.getenv("RUN_HF_SMOKE", "").lower() in {"1", "true", "yes"}


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_HF_SMOKE,
    reason="Set RUN_HF_SMOKE=1 to enable Hugging Face model-load smoke tests.",
)
def test_model_registry_loads_public_seq2seq_model() -> None:
    """Load a tiny public model to validate end-to-end model registry wiring."""
    model_id = os.getenv("HF_SMOKE_MODEL", "hf-internal-testing/tiny-random-t5")
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    cfg = Config(
        hf_model=model_id,
        hf_token=token,
        device_map=None,
        load_in_8bit=False,
        torch_dtype="float32",
    )
    model, tokenizer = load_base_model(cfg)
    assert model.config.is_encoder_decoder is True
    assert tokenizer is not None
