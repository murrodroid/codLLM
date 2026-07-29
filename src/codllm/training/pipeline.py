from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import Trainer

import codllm.wandb_utils as wandb_utils
from codllm import run_markers
from codllm.config import Config
from codllm.data import (
    DataHandler,
    DataSplits,
    prepare_sequence_classification_dataset,
    prepare_training_dataset,
    resolve_training_frames,
)
from codllm.run_directory import prepare_run_output_dir
from codllm.training.metadata import build_data_metadata, dataset_row_count
from codllm.training.model_setup import (
    initialize_training_components,
    resolve_classifier_label_space,
)
from codllm.training.stages import (
    TrainingStage,
    build_finetune_stage,
    build_pretraining_stage,
    build_train_stage,
    release_stage_trainer_memory,
)
from codllm.training.trainer_factory import run_training_stage
from codllm.training.visualizations import log_data_visualizations
from codllm.uncertainty.end_of_training import run_end_of_training_uncertainty


def _log_progress(message: str) -> None:
    """Print a timestamped training progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rendered = f"[{timestamp}] {message}"
    print(rendered, flush=True)
    wandb_utils.log_wandb_text(rendered)


def _initialize_wandb_for_data_prep(cfg: Config) -> None:
    """Initialize W&B before expensive local data preparation starts."""
    report_to, run_name = wandb_utils.resolve_wandb_reporting(cfg)
    wandb_utils.log_wandb_run_metadata(
        cfg=cfg,
        report_to=report_to,
        run_name=run_name,
        metadata={
            "run": {
                "phase": "data_preparation",
                "status": "started",
            }
        },
    )


def _train_from_datasets(
    cfg: Config,
    stage: TrainingStage,
    train_ds: Any,
    eval_ds: Any | None = None,
    holdout_eval_ds: Any | None = None,
    run_data_metadata: dict[str, Any] | None = None,
    label2id: dict[str, int] | None = None,
    id2label: dict[int, str] | None = None,
) -> tuple[Trainer, Any]:
    """Run one fine-tuning job from prepared datasets."""
    _log_progress(f"Initializing model and tokenizer for stage '{stage.name}'.")
    model, tokenizer, disable_fp16 = initialize_training_components(
        cfg=cfg,
        label2id=label2id,
        id2label=id2label,
    )
    trainer = run_training_stage(
        cfg=cfg,
        stage=stage,
        model=model,
        tokenizer=tokenizer,
        disable_fp16=disable_fp16,
        train_ds=train_ds,
        eval_ds=eval_ds,
        holdout_eval_ds=holdout_eval_ds,
        run_data_metadata=run_data_metadata,
        label2id=label2id,
        id2label=id2label,
    )
    return trainer, tokenizer


def _train_with_pretraining(
    cfg: Config,
    pretrain_stage: TrainingStage,
    finetune_stage: TrainingStage,
    pretrain_ds: Any,
    train_ds: Any,
    eval_ds: Any | None = None,
    holdout_eval_ds: Any | None = None,
    run_data_metadata: dict[str, Any] | None = None,
    label2id: dict[str, int] | None = None,
    id2label: dict[int, str] | None = None,
) -> tuple[Trainer, Any]:
    """Run optional pretraining first, then continue with regular fine-tuning."""
    _log_progress("Initializing model and tokenizer for pretraining flow.")
    model, tokenizer, disable_fp16 = initialize_training_components(
        cfg=cfg,
        label2id=label2id,
        id2label=id2label,
    )
    pretrain_trainer = run_training_stage(
        cfg=cfg,
        stage=pretrain_stage,
        model=model,
        tokenizer=tokenizer,
        disable_fp16=disable_fp16,
        train_ds=pretrain_ds,
        eval_ds=eval_ds,
        holdout_eval_ds=None,
        run_data_metadata=run_data_metadata,
        label2id=label2id,
        id2label=id2label,
    )
    release_stage_trainer_memory(cfg=cfg, trainer=pretrain_trainer)
    # Drop the caller's own reference so gc inside release can actually collect
    # the pretraining trainer (otherwise it stays alive until this function
    # returns, pinning its CUDA allocations through the finetune stage).
    pretrain_trainer = None
    trainer = run_training_stage(
        cfg=cfg,
        stage=finetune_stage,
        model=model,
        tokenizer=tokenizer,
        disable_fp16=disable_fp16,
        train_ds=train_ds,
        eval_ds=eval_ds,
        holdout_eval_ds=holdout_eval_ds,
        run_data_metadata=run_data_metadata,
        label2id=label2id,
        id2label=id2label,
    )
    return trainer, tokenizer


def evaluate_test_split(
    cfg: Config,
    trainer: Any,
    tokenizer: Any,
    test_ds: Any,
    label2id: dict[str, int] | None = None,
    metric_key_prefix: str = "test",
) -> dict[str, float] | None:
    """Run final evaluation on one split and emit metrics with the requested prefix."""
    if dataset_row_count(test_ds) in (None, 0):
        return None
    if not hasattr(trainer, "evaluate"):
        return None

    if cfg.model_task == "sequence_classification":
        if label2id is None:
            raise ValueError(
                "Sequence-classification test evaluation requires label mappings."
            )
        processed_test_ds = prepare_sequence_classification_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            dataset=test_ds,
            label2id=label2id,
        )
    else:
        target_max_length = cfg.resolved_max_target_length()
        processed_test_ds = prepare_training_dataset(
            cfg,
            tokenizer,
            test_ds,
            target_max_length,
        )
    # transformers caches eval dataloaders under the literal key "eval" for any
    # non-str dataset; with dataloader_persistent_workers=True this makes
    # evaluate(eval_dataset=test/holdout) silently reuse the cached VAL dataloader
    # and re-score val (the test/* == val/* and holdout_full/* == val/* aliasing).
    # Evict the stale entry so HF rebuilds a dataloader over the real split.
    getattr(trainer, "_eval_dataloaders", {}).pop("eval", None)
    raw_metrics = trainer.evaluate(
        eval_dataset=processed_test_ds,
        metric_key_prefix=metric_key_prefix,
    )
    return {key: float(value) for key, value in raw_metrics.items()}


def _source_count(dataset: Any, source_id: str) -> int | None:
    """Return count of rows from one source when source metadata is available."""
    if dataset is None:
        return 0
    if not hasattr(dataset, "columns"):
        return None
    if "source_id" not in dataset.columns:
        return None
    return int((dataset["source_id"].fillna("").astype(str) == source_id).sum())


def validate_split_source_integrity(cfg: Config, splits: DataSplits) -> None:
    """Fail fast when source-partitioned splits violate configured source rules."""
    forbidden_sources = {
        source_id.strip()
        for source_id in cfg.train_excluded_source_ids
        if source_id.strip()
    }
    if cfg.hold_out_dataset is not None:
        forbidden_sources.add(cfg.hold_out_dataset)
    if not forbidden_sources:
        return

    errors: list[str] = []
    for split_name, dataset in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        for source_id in sorted(forbidden_sources):
            count = _source_count(dataset, source_id)
            if count is None and dataset_row_count(dataset) not in (None, 0):
                errors.append(
                    f"{split_name} split is missing source_id metadata needed "
                    f"to validate source '{source_id}'"
                )
            elif count and count > 0:
                errors.append(
                    f"{split_name} split contains {count} row(s) from forbidden "
                    f"source '{source_id}'"
                )

    if cfg.hold_out_dataset is not None:
        holdout_count = _source_count(splits.holdout, cfg.hold_out_dataset)
        if holdout_count is None:
            errors.append("holdout split is missing source_id metadata")
        elif holdout_count <= 0:
            errors.append(
                f"holdout split contains no rows from '{cfg.hold_out_dataset}'"
            )
        if splits.holdout is not None and hasattr(splits.holdout, "columns"):
            if "source_id" in splits.holdout.columns:
                source_values = {
                    source
                    for source in splits.holdout["source_id"]
                    .fillna("")
                    .astype(str)
                    .tolist()
                    if source
                }
                unexpected_sources = sorted(
                    source_values.difference({cfg.hold_out_dataset})
                )
                if unexpected_sources:
                    errors.append(
                        "holdout split contains unexpected source_id values: "
                        + ", ".join(unexpected_sources)
                    )

    if errors:
        raise ValueError("Invalid source-partitioned splits: " + "; ".join(errors))


def train(
    cfg: Config,
    data_handler: DataHandler | None = None,
    force_reprocess: bool = False,
) -> tuple[Trainer, Any, DataSplits]:
    """Build or load data splits and launch training."""
    prepare_run_output_dir(cfg)
    # Clear any resume request left by a previous slot so its presence after
    # training unambiguously reflects *this* slot's outcome (the time-budget
    # callback re-writes it if the budget triggers again).
    state_dir = run_markers.run_state_dir(cfg.output_dir)
    run_markers.clear_resume_needed(state_dir)
    _initialize_wandb_for_data_prep(cfg)
    handler = data_handler or DataHandler(cfg)
    _log_progress("Preparing data splits.")
    splits = handler.get_splits(force_reprocess=force_reprocess)
    _log_progress(
        "Prepared data splits: "
        f"train={dataset_row_count(splits.train)}, "
        f"val={dataset_row_count(splits.val)}, "
        f"test={dataset_row_count(splits.test)}."
    )
    validate_split_source_integrity(cfg, splits)
    train_ds, eval_ds = resolve_training_frames(splits)
    run_data_metadata = build_data_metadata(
        cfg=cfg,
        splits=splits,
        force_reprocess=force_reprocess,
        handler=handler,
    )
    classifier_label2id: dict[str, int] | None = None
    classifier_id2label: dict[int, str] | None = None
    if cfg.model_task == "sequence_classification":
        if cfg.max_label_count != 1:
            raise ValueError(
                "sequence_classification currently supports max_label_count=1 only."
            )
        classifier_label2id, classifier_id2label = resolve_classifier_label_space(
            handler=handler
        )
        run_data_metadata["classification"] = {
            "num_labels": len(classifier_label2id),
            "label_source": {
                "masterlist_path": str(Path(cfg.pretrain_masterlist_path).resolve()),
                "sheet_name": cfg.pretrain_masterlist_sheet_name,
                "column": "ICD10h",
            },
        }

    masterlist_inject_metrics_loader = getattr(
        handler,
        "get_masterlist_inject_metrics",
        None,
    )
    masterlist_inject_metrics = (
        masterlist_inject_metrics_loader()
        if callable(masterlist_inject_metrics_loader)
        else None
    )
    if masterlist_inject_metrics is not None:
        run_data_metadata["masterlist_injection"] = masterlist_inject_metrics
    training_balance_metrics_loader = getattr(
        handler,
        "get_training_balance_metrics",
        None,
    )
    training_balance_metrics = (
        training_balance_metrics_loader()
        if callable(training_balance_metrics_loader)
        else None
    )
    if training_balance_metrics is not None:
        run_data_metadata["training_balance"] = training_balance_metrics

    pretrain_loader = getattr(handler, "get_pretraining_train_dataframe", None)
    if callable(pretrain_loader):
        _log_progress("Preparing optional pretraining dataset.")
    pretrain_ds = pretrain_loader() if callable(pretrain_loader) else None
    if pretrain_ds is not None:
        _log_progress(f"Prepared pretraining dataset: train={int(len(pretrain_ds))}.")
    pretrain_upsampling_metrics_loader = getattr(
        handler,
        "get_pretraining_upsampling_metrics",
        None,
    )
    pretrain_upsampling_metrics = (
        pretrain_upsampling_metrics_loader()
        if callable(pretrain_upsampling_metrics_loader)
        else None
    )
    pretrain_multicod_metrics_loader = getattr(
        handler,
        "get_pretraining_multicod_metrics",
        None,
    )
    pretrain_multicod_metrics = (
        pretrain_multicod_metrics_loader()
        if callable(pretrain_multicod_metrics_loader)
        else None
    )
    log_data_visualizations(
        cfg=cfg,
        splits=splits,
        balance_metrics=training_balance_metrics,
        masterlist_inject_metrics=masterlist_inject_metrics,
        pretraining_upsampling_metrics=pretrain_upsampling_metrics,
        pretraining_multicod_metrics=pretrain_multicod_metrics,
    )
    if cfg.pretrain_enabled and pretrain_loader is None:
        raise AttributeError(
            "Configured data_handler does not support pretraining datasets."
        )
    if cfg.pretrain_enabled and pretrain_ds is None:
        raise ValueError(
            "Pretraining is enabled but no pretraining dataset was returned."
        )
    if cfg.pretrain_enabled and pretrain_ds is not None:
        if eval_ds is None:
            raise ValueError(
                "Pretraining requires a non-empty validation split from the regular dataset."
            )
        pretrain_stage = build_pretraining_stage(cfg)
        finetune_stage = build_finetune_stage(cfg)
        run_data_metadata["pretraining"] = {
            "enabled": True,
            "masterlist_path": str(Path(cfg.pretrain_masterlist_path).resolve()),
            "sheet_name": cfg.pretrain_masterlist_sheet_name,
            "train_rows": int(len(pretrain_ds)),
            "num_train_epochs": pretrain_stage.num_train_epochs,
            "learning_rate": pretrain_stage.learning_rate,
            "warmup_ratio": pretrain_stage.warmup_ratio,
            "eval_every_n_epochs": pretrain_stage.eval_every_n_epochs,
            "lr_scheduler_type": pretrain_stage.lr_scheduler_type,
        }
        if pretrain_upsampling_metrics is not None:
            run_data_metadata["pretraining"]["upsampling"] = pretrain_upsampling_metrics
        if pretrain_multicod_metrics is not None:
            run_data_metadata["pretraining"]["multicod_synthetic"] = (
                pretrain_multicod_metrics
            )
        trainer, tokenizer = _train_with_pretraining(
            cfg=cfg,
            pretrain_stage=pretrain_stage,
            finetune_stage=finetune_stage,
            pretrain_ds=pretrain_ds,
            train_ds=train_ds,
            eval_ds=eval_ds,
            holdout_eval_ds=splits.holdout_eval,
            run_data_metadata=run_data_metadata,
            label2id=classifier_label2id,
            id2label=classifier_id2label,
        )
    else:
        trainer, tokenizer = _train_from_datasets(
            cfg=cfg,
            stage=build_train_stage(cfg),
            train_ds=train_ds,
            eval_ds=eval_ds,
            holdout_eval_ds=splits.holdout_eval,
            run_data_metadata=run_data_metadata,
            label2id=classifier_label2id,
            id2label=classifier_id2label,
        )
    state_dir = run_markers.run_state_dir(cfg.output_dir)
    if run_markers.is_resume_needed(state_dir):
        # Wall-time budget stopped training before it finished. A checkpoint is
        # saved; skip the one-shot final evaluation and let the LSF script
        # resubmit for another slot. Exit cleanly (status 0) so the run reads as
        # "needs resume", not "crashed".
        _log_progress(
            "Time budget reached before training finished; checkpoint saved and "
            f"'{run_markers.RESUME_NEEDED_MARKER}' set. Skipping final "
            "evaluation; the job will be resubmitted to continue."
        )
        wandb_utils.log_run_status(cfg, "needs_resume")
        wandb_utils.finish_wandb_run(cfg)
        return trainer, tokenizer, splits

    _run_final_evaluation(
        cfg=cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        splits=splits,
        label2id=classifier_label2id,
    )
    wandb_utils.log_optimization_result(cfg, trainer)
    # Training truly finished (epochs exhausted or early stopping) and final
    # evaluation ran exactly once. Record completion so a stray resubmission
    # does not re-run training or evaluation.
    run_markers.mark_training_complete(state_dir)
    wandb_utils.log_run_status(cfg, "complete")
    wandb_utils.finish_wandb_run(cfg)
    return trainer, tokenizer, splits


def _run_final_evaluation(
    cfg: Config,
    trainer: Any,
    tokenizer: Any,
    splits: DataSplits,
    label2id: dict[str, int] | None = None,
) -> None:
    """Run the one-shot final test, full-holdout, and uncertainty evaluation."""
    if cfg.final_test_eval_enabled:
        evaluate_test_split(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            test_ds=splits.test,
            label2id=label2id,
        )
    if dataset_row_count(splits.holdout) not in (None, 0):
        evaluate_test_split(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            test_ds=splits.holdout,
            label2id=label2id,
            metric_key_prefix="holdout_full",
        )
    if (
        cfg.final_test_eval_enabled
        and cfg.uncertainty_eval
        and cfg.model_task == "seq2seq"
        and dataset_row_count(splits.test) not in (None, 0)
    ):
        _log_progress("Running end-of-training uncertainty pass on test split.")
        try:
            run_end_of_training_uncertainty(
                cfg=cfg,
                model=trainer.model,
                tokenizer=tokenizer,
                test_df=splits.test,
                val_df=splits.val,
                run_dir=Path(cfg.output_dir),
            )
        except Exception as exc:  # pragma: no cover - defensive: never fail training
            _log_progress(f"Uncertainty pass skipped due to error: {exc!r}")
