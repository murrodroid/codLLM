import netrc
import os
from pathlib import Path
import warnings

from codllm.config import Config


def has_wandb_credentials() -> bool:
    """Return True when W&B credentials are available via env or netrc."""
    if os.getenv("WANDB_API_KEY"):
        return True

    for filename in (".netrc", "_netrc"):
        netrc_path = Path.home() / filename
        if not netrc_path.exists():
            continue
        try:
            auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
        except (OSError, netrc.NetrcParseError):
            continue
        if auth and auth[2]:
            return True
    return False


def resolve_wandb_reporting(cfg: Config) -> tuple[str | list[str], str | None]:
    """Resolve Trainer reporting settings and runtime environment for W&B."""
    wandb_cfg = cfg.wandb

    if not wandb_cfg.enabled or wandb_cfg.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return "none", None

    os.environ.setdefault("WANDB_PROJECT", wandb_cfg.project)
    if wandb_cfg.entity:
        os.environ["WANDB_ENTITY"] = wandb_cfg.entity

    if wandb_cfg.mode in ("online", "offline"):
        os.environ["WANDB_MODE"] = wandb_cfg.mode
        return ["wandb"], wandb_cfg.run_name

    if has_wandb_credentials():
        return ["wandb"], wandb_cfg.run_name

    os.environ["WANDB_MODE"] = "disabled"
    warnings.warn(
        (
            "W&B is enabled but no WANDB_API_KEY or ~/.netrc credentials were found. "
            "Falling back to report_to='none'."
        ),
        stacklevel=2,
    )
    return "none", None
