from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import Trainer

import codllm.wandb_utils as wandb_utils
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


def _log_progress(message: str) -> None:
    """Print a timestamped training progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
    raw_metrics = trainer.evaluate(
        eval_dataset=processed_test_ds,
        metric_key_prefix=metric_key_prefix,
    )
    return {key: float(value) for key, value in raw_metrics.items()}


def train(
    cfg: Config,
    data_handler: DataHandler | None = None,
    force_reprocess: bool = False,
) -> tuple[Trainer, Any, DataSplits]:
    """Build or load data splits and launch training."""
    prepare_run_output_dir(cfg)
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
    evaluate_test_split(
        cfg=cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        test_ds=splits.test,
        label2id=classifier_label2id,
    )
    if dataset_row_count(splits.holdout) not in (None, 0):
        evaluate_test_split(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            test_ds=splits.holdout,
            label2id=classifier_label2id,
            metric_key_prefix="holdout_test",
        )
    return trainer, tokenizer, splits
