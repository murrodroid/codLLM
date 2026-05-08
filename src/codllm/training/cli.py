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
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Resume the latest run-* dir under output_dir when one exists; "
            "pair with --per-size-output-dir on size-sweep HPC jobs so each "
            "model has its own stable resume target."
        ),
    )
    parser.add_argument(
        "--per-size-output-dir",
        action="store_true",
        help=(
            "Insert the model-name slug below the configured output_dir, so "
            "flan-t5-small/base/large/xl runs get separate, resume-stable "
            "run roots."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override cfg.output_dir. Pointing at an existing run-* dir "
            "reuses it (use this for explicit resume targets)."
        ),
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=None,
        help=(
            "Wall-time budget in seconds; the trainer saves and stops "
            "before the budget minus the safety margin elapses."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "Number of evals without improvement before stopping. Requires "
            "either save_strategy='best' or load_best_model_at_end=True."
        ),
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
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.auto_resume:
        cfg.auto_resume = True
    if args.per_size_output_dir:
        cfg.per_size_output_dir = True
    if args.max_runtime_seconds is not None:
        cfg.max_runtime_seconds = args.max_runtime_seconds
    if args.early_stopping_patience is not None:
        cfg.early_stopping_patience = args.early_stopping_patience
    _, _, splits = train(
        cfg=cfg,
        force_reprocess=args.force_reprocess,
    )
    print(
        f"Training completed. train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}."
    )
