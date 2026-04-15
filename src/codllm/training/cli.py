import argparse

from codllm.config import config_from_env
from codllm.training.pipeline import train


def parse_args() -> argparse.Namespace:
    """Parse CLI args for running training from the command line."""
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Rebuild processed data even when a processed file already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed for reproducible runs.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Override config data seed for split/sampler reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch training using the default project config."""
    args = parse_args()
    cfg = config_from_env()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_seed is not None:
        cfg.data_seed = args.data_seed
    _, _, splits = train(
        cfg=cfg,
        force_reprocess=args.force_reprocess,
    )
    print(
        f"Training completed. train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}."
    )
