import os
import random

import torch

from codllm.config import Config


def _seed_numpy(seed: int) -> None:
    """Seed NumPy when available."""
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def _set_deterministic_algorithms(enabled: bool, warn_only: bool) -> None:
    """Set torch deterministic algorithms with compatibility fallback."""
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    except TypeError:
        torch.use_deterministic_algorithms(enabled)


def configure_reproducibility(cfg: Config) -> None:
    """Apply deterministic and seed settings from config."""
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))

    random.seed(cfg.seed)
    _seed_numpy(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    _set_deterministic_algorithms(
        cfg.deterministic_algorithms, cfg.deterministic_algorithms_warn_only
    )
