import random

import pytest

from codllm.config import Config
from codllm.runtime.reproducibility import configure_reproducibility
import codllm.runtime.reproducibility as reproducibility_module


def test_configure_reproducibility_resets_python_random() -> None:
    """Repeated setup with same seed should reproduce python random values."""
    cfg = Config(seed=11)
    configure_reproducibility(cfg)
    first = random.random()
    configure_reproducibility(cfg)
    second = random.random()
    assert first == second


def test_configure_reproducibility_sets_torch_determinism_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch deterministic algorithm settings should come from config."""
    captured: dict[str, bool] = {}

    def fake_use_deterministic_algorithms(
        enabled: bool, warn_only: bool = False
    ) -> None:
        captured["enabled"] = enabled
        captured["warn_only"] = warn_only

    monkeypatch.setattr(
        reproducibility_module.torch,
        "use_deterministic_algorithms",
        fake_use_deterministic_algorithms,
    )
    cfg = Config(
        deterministic_algorithms=True,
        deterministic_algorithms_warn_only=False,
    )
    configure_reproducibility(cfg)
    assert captured["enabled"] is True
    assert captured["warn_only"] is False
