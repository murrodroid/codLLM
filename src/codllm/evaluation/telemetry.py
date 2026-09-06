"""Best-effort scalar reporting after private evaluation artifacts are safely persisted."""

from __future__ import annotations

import logging
import math
from typing import Any

from codllm import wandb_utils
from codllm.config import Config


def log_completed_evaluation(
    cfg: Config, results: dict[str, dict[str, float]], metadata: dict[str, Any]
) -> None:
    """Mirror finite metrics to W&B without uploading raw records or risking local results."""
    if not cfg.wandb.enabled:
        return
    try:
        report_to, run_name = wandb_utils.resolve_wandb_reporting(cfg)
        if not wandb_utils.report_to_includes_wandb(report_to):
            return
        wandb_utils.log_wandb_run_metadata(cfg, report_to, run_name, metadata)
        wandb = wandb_utils._import_wandb()
        if wandb.run is not None:
            metrics = {
                f"{scope}/{key}": value
                for scope, values in results.items()
                for key, value in values.items()
                if math.isfinite(value)
            }
            wandb.log(metrics)
            wandb.run.summary.update(metrics)
            wandb_utils.log_run_status(cfg, "complete")
    except Exception:
        logging.getLogger(__name__).exception(
            "W&B evaluation sync failed; private local results are intact."
        )
    finally:
        wandb_utils.finish_wandb_run(cfg)
