from datetime import datetime
from typing import Any

from pathlib import Path

from transformers import (
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
)

import codllm.wandb_utils as wandb_utils
from codllm.config import Config
from codllm.data import (
    prepare_sequence_classification_dataset,
    prepare_training_dataset,
)
from codllm.metrics import (
    build_exact_match_accuracy_metric,
    build_sequence_classification_metric,
    collect_input_strings,
    collect_label_classes,
)
from codllm.trainer_logging import (
    EvaluateEveryNEpochsCallback,
    HoldoutEvaluationCallback,
    SigtermSaveCallback,
    StageScopedSeq2SeqTrainer,
    StageScopedTrainer,
    TimeBudgetCallback,
)
from codllm.training.arguments import build_training_args
from codllm.training.metadata import (
    build_stage_run_metadata,
    print_training_configuration,
)
from codllm.training.stages import TrainingStage, should_apply_eval_interval_callback
from codllm.training.visualizations import MetricArtifactLogger


def _log_progress(message: str) -> None:
    """Print a timestamped training progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_training_stage(
    cfg: Config,
    stage: TrainingStage,
    model: Any,
    tokenizer: Any,
    disable_fp16: bool,
    train_ds: Any,
    eval_ds: Any | None = None,
    holdout_eval_ds: Any | None = None,
    run_data_metadata: dict[str, Any] | None = None,
    label2id: dict[str, int] | None = None,
    id2label: dict[int, str] | None = None,
) -> Trainer:
    """Preprocess datasets and run one training stage."""
    target_max_length = cfg.resolved_max_target_length()
    _log_progress(f"Tokenizing datasets for stage '{stage.name}'.")
    if cfg.model_task == "sequence_classification":
        if label2id is None or id2label is None:
            raise ValueError(
                "Sequence-classification training requires label mappings."
            )
        processed_train_ds = prepare_sequence_classification_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            dataset=train_ds,
            label2id=label2id,
        )
        processed_eval_ds = None
        if eval_ds is not None:
            processed_eval_ds = prepare_sequence_classification_dataset(
                cfg=cfg,
                tokenizer=tokenizer,
                dataset=eval_ds,
                label2id=label2id,
            )
        processed_holdout_eval_ds = None
        if holdout_eval_ds is not None:
            processed_holdout_eval_ds = prepare_sequence_classification_dataset(
                cfg=cfg,
                tokenizer=tokenizer,
                dataset=holdout_eval_ds,
                label2id=label2id,
            )
        collator: Any = DataCollatorWithPadding(tokenizer=tokenizer)
        train_input_strings = collect_input_strings(processed_train_ds, tokenizer)
        train_classes = None
    else:
        processed_train_ds = prepare_training_dataset(
            cfg,
            tokenizer,
            train_ds,
            target_max_length,
        )
        processed_eval_ds = None
        if eval_ds is not None:
            processed_eval_ds = prepare_training_dataset(
                cfg,
                tokenizer,
                eval_ds,
                target_max_length,
            )
        processed_holdout_eval_ds = None
        if holdout_eval_ds is not None:
            processed_holdout_eval_ds = prepare_training_dataset(
                cfg,
                tokenizer,
                holdout_eval_ds,
                target_max_length,
            )
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        train_classes = collect_label_classes(
            train_ds,
            label_column=cfg.dataset_label_column,
            label_separator=cfg.label_separator,
        )
        train_input_strings = collect_input_strings(processed_train_ds, tokenizer)
    _log_progress(f"Finished tokenizing datasets for stage '{stage.name}'.")

    stage_run_data_metadata = build_stage_run_metadata(
        run_data_metadata=run_data_metadata,
        stage=stage,
        train_ds=train_ds,
        eval_ds=eval_ds,
    )
    args = build_training_args(
        cfg=cfg,
        has_eval=processed_eval_ds is not None,
        stage=stage,
        generation_max_length=target_max_length,
        disable_fp16=disable_fp16,
        include_inputs_for_metrics=(
            processed_eval_ds is not None or processed_holdout_eval_ds is not None
        ),
    )
    wandb_utils.patch_transformers_wandb_log_rewrite(report_to=args.report_to, cfg=cfg)
    training_args_payload = args.to_dict() if hasattr(args, "to_dict") else None
    metadata_payload = wandb_utils.build_experiment_metadata(
        cfg=cfg,
        data_metadata=stage_run_data_metadata,
        training_args=training_args_payload,
    )
    wandb_utils.log_wandb_run_metadata(
        cfg=cfg,
        report_to=args.report_to,
        run_name=args.run_name,
        metadata=metadata_payload,
    )
    print_training_configuration(cfg, args, stage_run_data_metadata)
    metric_artifact_logger = MetricArtifactLogger(cfg)

    callbacks: list[TrainerCallback] = []
    eval_strategy_value = (
        args.eval_strategy.value
        if hasattr(args.eval_strategy, "value")
        else str(args.eval_strategy)
    )
    if should_apply_eval_interval_callback(
        stage=stage,
        eval_strategy_value=eval_strategy_value,
        has_eval_dataset=processed_eval_ds is not None,
    ):
        callbacks.append(EvaluateEveryNEpochsCallback(stage.eval_every_n_epochs))
    holdout_callback = None
    if processed_holdout_eval_ds is not None and cfg.hold_out_evaluate_per is not None:
        holdout_callback = HoldoutEvaluationCallback(
            evaluate_per=cfg.hold_out_evaluate_per,
            baseline_eval_dataset=processed_eval_ds,
            eval_dataset=processed_holdout_eval_ds,
            eval_steps=cfg.eval_steps,
        )
        callbacks.append(holdout_callback)

    if cfg.early_stopping_patience > 0 and processed_eval_ds is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        )

    if cfg.max_runtime_seconds > 0:
        callbacks.append(
            TimeBudgetCallback(
                max_runtime_seconds=float(cfg.max_runtime_seconds),
                safety_margin_seconds=float(cfg.runtime_safety_margin_seconds),
            )
        )

    callbacks.append(SigtermSaveCallback())

    if cfg.model_task == "sequence_classification":
        trainer = StageScopedTrainer(
            model=model,
            args=args,
            train_dataset=processed_train_ds,
            eval_dataset=processed_eval_ds,
            data_collator=collator,
            processing_class=tokenizer,
            stage_name=stage.name,
            callbacks=callbacks or None,
            compute_metrics=(
                build_sequence_classification_metric(
                    id2label=id2label,
                    tokenizer=tokenizer,
                    train_input_strings=train_input_strings or None,
                    artifact_logger=metric_artifact_logger,
                    metric_mode=cfg.wandb.metric_mode,
                    save_metric=cfg.save_strategy_best_metric,
                )
                if (
                    (
                        processed_eval_ds is not None
                        or processed_holdout_eval_ds is not None
                    )
                    and id2label is not None
                )
                else None
            ),
        )
    else:
        trainer = StageScopedSeq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=processed_train_ds,
            eval_dataset=processed_eval_ds,
            data_collator=collator,
            processing_class=tokenizer,
            stage_name=stage.name,
            callbacks=callbacks or None,
            compute_metrics=(
                build_exact_match_accuracy_metric(
                    tokenizer,
                    label_separator=cfg.label_separator,
                    max_label_count=cfg.max_label_count,
                    train_classes=train_classes or None,
                    train_input_strings=train_input_strings or None,
                    artifact_logger=metric_artifact_logger,
                    metric_mode=cfg.wandb.metric_mode,
                    save_metric=cfg.save_strategy_best_metric,
                )
                if processed_eval_ds is not None
                or processed_holdout_eval_ds is not None
                else None
            ),
        )

    if holdout_callback is not None:
        holdout_callback.attach_trainer(trainer)

    resume_arg: bool | str = False
    if cfg.auto_resume and _has_existing_checkpoint(args.output_dir):
        resume_arg = True
        _log_progress(
            f"Auto-resume: existing checkpoint found in '{args.output_dir}'."
        )
    trainer.train(resume_from_checkpoint=resume_arg)
    return trainer


def _has_existing_checkpoint(output_dir: str) -> bool:
    """Return True when ``output_dir`` already contains a Trainer checkpoint."""
    base = Path(output_dir)
    if not base.exists():
        return False
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            return True
    return False
