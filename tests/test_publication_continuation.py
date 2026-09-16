"""Offline guards, identity isolation, and real CPU resume tests for publication continuations."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from codllm.config import Config, WandbConfig, config_from_env
from codllm import run_markers
from codllm.evaluation.artifacts import training_recipe, write_json
from codllm.evaluation.provenance import content_digest
from codllm.trainer_logging import StageScopedTrainer, rewrite_metric_key_for_stage
from codllm.training.continuation import (
    configuration_payload,
    inspect_parent,
    prepare_continuation,
    validate_continuation,
)
from codllm.training.continuation_trainer import (
    ContinuationTrainer,
    ContinuationSigtermSaveCallback,
    ResumableEarlyStoppingCallback,
)


def _parent_metadata(cfg: Config, checkpoint: Path, state_dir: Path) -> None:
    """Create synthetic frozen parent metadata around a fixture checkpoint."""
    publication = Path(cfg.output_dir) / "publication"
    (publication / "manifests").mkdir(parents=True)
    (publication / "manifests/fixture.parquet").write_bytes(b"synthetic fixture")
    write_json(publication / "reference.json", {"fixture": True})
    write_json(publication / "effective_config.json", configuration_payload(cfg))
    write_json(
        publication / "contract.json",
        {
            "version": 1,
            "manifests": {},
            "protocol": "cod",
            "training_recipe_digest": content_digest(
                training_recipe(configuration_payload(cfg))
            ),
        },
    )
    state_dir.mkdir()
    for name in (".training_complete", ".final_eval_done"):
        (state_dir / name).write_text("completed fixture\n")
    (state_dir / "wandb_run_id.txt").write_text("parent01\n")


@pytest.fixture
def continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Build a tiny isolated parent and its requested patience-only continuation."""
    for key in (
        "WANDB_RUN_ID",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "CODLLM_RUN_STATE_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    parent = Config(
        output_dir=str(tmp_path / "parent/run-1"),
        early_stopping_patience=10,
        num_train_epochs=120,
        auto_resume=True,
        publication_eval_enabled=True,
        prediction_export_enabled=True,
        final_test_eval_enabled=False,
        evaluation_protocol="cod",
        device=torch.device("cpu"),
    )
    checkpoint = Path(parent.output_dir) / "finetune/checkpoint-700"
    checkpoint.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "training_args.bin",
    ):
        (checkpoint / name).write_bytes(b"synthetic checkpoint fixture")
    write_json(
        checkpoint / "trainer_state.json",
        {
            "global_step": 700,
            "best_global_step": 700,
            "epoch": 70,
            "best_model_checkpoint": str(checkpoint),
            "best_metric": 0.6,
            "max_steps": 1200,
            "num_train_epochs": 120,
            "log_history": [{"epoch": 70}],
        },
    )
    state_dir = tmp_path / "parent-state"
    _parent_metadata(parent, checkpoint, state_dir)
    return replace(
        parent,
        output_dir=str(tmp_path / "child/run-0001"),
        early_stopping_patience=20,
        continuation_source_run_dir=parent.output_dir,
        continuation_source_state_dir=str(state_dir),
        continuation_parent_wandb_run="codllmdev/codllm/parent01",
    )


def test_copy_isolated_and_idempotent(continuation: Config) -> None:
    """Preparation copies full state, never mutates parents, and retains one child identity."""
    parent = Path(continuation.continuation_source_run_dir)
    before = {
        p.relative_to(parent): p.read_bytes() for p in parent.rglob("*") if p.is_file()
    }
    info = prepare_continuation(continuation)
    child = Path(continuation.output_dir)
    assert info == prepare_continuation(continuation)
    assert info["wandb_run_id"] != "parent01"
    assert info["original_max_steps"] == 1200
    assert info["target_epochs"] == 120
    assert info["target_patience"] == 20
    original = parent / "finetune/checkpoint-700/optimizer.pt"
    copied = child / "finetune/checkpoint-700/optimizer.pt"
    assert original.stat().st_ino != copied.stat().st_ino
    state = json.loads((copied.parent / "trainer_state.json").read_text())
    assert state["log_history"] == []
    assert state["best_model_checkpoint"] == str(copied.parent)
    assert not (child / ".training_complete").exists()
    assert before == {
        p.relative_to(parent): p.read_bytes() for p in parent.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "change",
    [
        {"num_train_epochs": 160},
        {"lr": 0.1},
        {"seed": 999},
        {"multicod_synthetic_ratio": 0.8},
        {"early_stopping_patience": 10},
        {"final_test_eval_enabled": True},
    ],
)
def test_reject_unrequested_changes(continuation: Config, change: dict) -> None:
    """Only increased patience may differ; data, schedule, ceiling, and test policy remain frozen."""
    with pytest.raises(ValueError):
        inspect_parent(replace(continuation, **change))


def test_parent_completion_and_destination_guards(continuation: Config) -> None:
    """Active parents and parent-as-destination errors fail before copying files."""
    with pytest.raises(ValueError, match="separate"):
        inspect_parent(
            replace(continuation, output_dir=continuation.continuation_source_run_dir)
        )
    (Path(continuation.continuation_source_state_dir) / ".training_complete").unlink()
    with pytest.raises(ValueError, match="not completed"):
        prepare_continuation(continuation)
    assert not Path(continuation.output_dir).exists()


def test_missing_state_and_tampered_lineage_rejected(continuation: Config) -> None:
    """Missing prepared checkpoints and altered branch settings must never start fresh training."""
    with pytest.raises(ValueError, match="missing"):
        validate_continuation(continuation)
    prepare_continuation(continuation)
    manifest = Path(continuation.output_dir) / "continuation.json"
    info = json.loads(manifest.read_text())
    info["target_patience"] = 99
    write_json(manifest, info)
    with pytest.raises(ValueError, match="modified"):
        validate_continuation(continuation)


def test_foreign_wandb_identity_rejected(
    continuation: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent identity in the environment cannot overwrite the original run."""
    prepare_continuation(continuation)
    monkeypatch.setenv("WANDB_RUN_ID", "parent01")
    with pytest.raises(ValueError, match="WANDB_RUN_ID"):
        validate_continuation(continuation)


def test_legacy_recipe_digest_unchanged() -> None:
    """New operational fields must not invalidate frozen original-run contracts."""
    current = configuration_payload(Config())
    legacy = {k: v for k, v in current.items() if not k.startswith("continuation_")}
    assert training_recipe(current) == training_recipe(legacy)


def test_missing_optimizer_refused_before_training(continuation: Config) -> None:
    """Do not let Transformers silently fall back to a fresh optimizer."""
    prepare_continuation(continuation)
    (Path(continuation.output_dir) / "finetune/checkpoint-700/optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete resumable"):
        validate_continuation(continuation)


def test_sigterm_marks_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A graceful scheduler interruption must never masquerade as completed training."""
    monkeypatch.setenv("CODLLM_RUN_STATE_DIR", str(tmp_path / "state"))
    callback = ContinuationSigtermSaveCallback()
    callback._signaled = True
    callback.on_step_end(
        SimpleNamespace(output_dir=str(tmp_path / "run-1/finetune")),
        TrainerState(),
        TrainerControl(),
    )
    assert run_markers.is_resume_needed(tmp_path / "state")
    assert not run_markers.is_training_complete(tmp_path / "state")


def test_env_and_selected_metric_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continuation settings parse centrally and selected-model scores do not extend training curves."""
    monkeypatch.setenv("CODLLM_CONTINUATION_SOURCE_RUN_DIR", "checkpoints/parent/run-1")
    monkeypatch.setenv("CODLLM_CONTINUATION_LR_SCHEDULE", "original")
    cfg = config_from_env()
    assert cfg.continuation_source_run_dir == "checkpoints/parent/run-1"
    assert cfg.continuation_lr_schedule == "original"
    assert (
        rewrite_metric_key_for_stage("selected_val_micro_f1", "finetune")
        == "selected/val/micro_f1"
    )


def test_callback_restores_count_not_old_patience() -> None:
    """A resumed allocation preserves elapsed patience without restoring the old limit."""
    callback = ResumableEarlyStoppingCallback(early_stopping_patience=20)
    state = TrainerState(
        stateful_callbacks={
            "EarlyStoppingCallback": {
                "args": {"early_stopping_patience": 10},
                "attributes": {"early_stopping_patience_counter": 13},
            }
        }
    )
    args = SimpleNamespace(
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        eval_strategy="epoch",
    )
    callback.on_train_begin(args, state, TrainerControl())
    assert callback.early_stopping_patience_counter == 13
    assert callback.early_stopping_patience == 20


def test_real_cpu_continuation_across_pause(tmp_path: Path) -> None:
    """Restore optimizer/LR/epoch state and retain patience across a mid-epoch HPC-like pause."""
    torch.set_num_threads(1)
    model_config = BertConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
    )
    dataset = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": i % 2}
        for i in range(4)
    ]

    def arguments(path: Path) -> TrainingArguments:
        """Build short CPU-only arguments with two updates per epoch."""
        return TrainingArguments(
            output_dir=str(path),
            use_cpu=True,
            num_train_epochs=6,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            eval_strategy="epoch",
            save_strategy="best",
            metric_for_best_model="macro_f1",
            load_best_model_at_end=True,
            save_total_limit=1,
            report_to="none",
            learning_rate=1e-3,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            disable_tqdm=True,
            dataloader_num_workers=0,
        )

    def metrics(prediction: object) -> dict[str, float]:
        """Use a constant score to make the exact patience boundary testable."""
        return {"macro_f1": 0.5}

    parent_root = tmp_path / "parent/run-1"
    parent = StageScopedTrainer(
        model=BertForSequenceClassification(model_config),
        args=arguments(parent_root / "finetune"),
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    parent.train()
    assert parent.state.epoch == 2
    parent_cfg = Config(
        output_dir=str(parent_root),
        num_train_epochs=6,
        early_stopping_patience=1,
        auto_resume=True,
        publication_eval_enabled=True,
        prediction_export_enabled=True,
        final_test_eval_enabled=False,
        device=torch.device("cpu"),
        wandb=WandbConfig(enabled=False),
    )
    _parent_metadata(
        parent_cfg, Path(parent.state.best_model_checkpoint), tmp_path / "parent-state"
    )
    cfg = replace(
        parent_cfg,
        output_dir=str(tmp_path / "child/run-0001"),
        early_stopping_patience=4,
        continuation_source_run_dir=str(parent_root),
        continuation_source_state_dir=str(tmp_path / "parent-state"),
        continuation_parent_wandb_run="codllmdev/codllm/parent01",
    )
    info = prepare_continuation(cfg)

    class Pause(TrainerCallback):
        """Force a wall-time-like pause halfway through epoch three."""

        def on_step_end(self, args, state, control, **kwargs):
            """Save a resumable checkpoint at update five."""
            if state.global_step == 5:
                control.should_save = True
                control.should_training_stop = True
                run_markers.mark_resume_needed(cfg.output_dir)
            return control

    def resumed(callbacks: list) -> ContinuationTrainer:
        """Create a fresh process-equivalent trainer for the same child branch."""
        trainer = ContinuationTrainer(
            model=BertForSequenceClassification(model_config),
            args=arguments(Path(cfg.output_dir) / "finetune"),
            train_dataset=dataset,
            eval_dataset=dataset,
            compute_metrics=metrics,
            callbacks=[
                ResumableEarlyStoppingCallback(early_stopping_patience=4),
                *callbacks,
            ],
        )
        trainer.continuation_manifest = info
        return trainer

    first = resumed([Pause()])
    first.train(resume_from_checkpoint=True)
    assert first.state.global_step == 5
    paused = Path(cfg.output_dir) / "finetune/checkpoint-5"
    state = json.loads((paused / "trainer_state.json").read_text())
    assert (
        state["stateful_callbacks"]["ResumableEarlyStoppingCallback"]["attributes"][
            "early_stopping_patience_counter"
        ]
        == 1
    )
    run_markers.clear_resume_needed(cfg.output_dir)
    second = resumed([])
    second.train(resume_from_checkpoint=True)
    assert second.state.global_step == 10
    assert second.state.epoch == 5
    assert second.lr_scheduler.lr_lambdas[0](12) == 0
    assert second.lr_scheduler.lr_lambdas[0](16) == 0
