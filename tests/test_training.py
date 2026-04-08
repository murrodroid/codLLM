import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

import codllm.metrics as metrics_module
import codllm.run_directory as run_directory_module
import codllm.trainer_logging as trainer_logging_module
import codllm.train as train_module
import codllm.wandb_utils as wandb_utils_module
from codllm.config import Config, WandbConfig
from codllm.data_handler import DataSplits
from codllm.preprocess import build_preprocess_fn
from codllm.train import build_training_args


class DummyTokenizer:
    """Tokenizer stub for preprocessing tests."""

    def __call__(
        self,
        texts: Optional[List[str]] = None,
        *,
        text_target: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> Dict[str, Any]:
        values = text_target if text_target is not None else texts
        assert values is not None
        assert truncation is True
        assert max_length is not None
        encoded = [[len(value)] for value in values]
        return {"input_ids": encoded}


class DummyDecodeTokenizer:
    """Tokenizer stub for metric decoding tests."""

    pad_token_id = 0
    token_map = {
        0: "<pad>",
        1: "A00",
        2: "A01",
        3: "B00",
    }

    def batch_decode(
        self, sequences: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode integer token ids into space-separated token strings."""
        decoded: List[str] = []
        for sequence in sequences:
            tokens = [
                self.token_map.get(int(token), str(int(token))) for token in sequence
            ]
            if skip_special_tokens:
                tokens = [token for token in tokens if token != "<pad>"]
            decoded.append(" ".join(tokens))
        return decoded


def test_preprocess_uses_configured_columns() -> None:
    """Preprocessing should read source and label columns from config."""
    cfg = Config(dataset_text_column="prompt", dataset_label_column="answer")
    preprocess = build_preprocess_fn(cfg, DummyTokenizer())
    batch = {"prompt": ["hello"], "answer": ["world"]}
    out = preprocess(batch)
    assert out["input_ids"] == [[5]]
    assert out["labels"] == [[5]]


def test_preprocess_raises_when_column_missing() -> None:
    """Preprocessing should fail fast when expected columns are missing."""
    cfg = Config(dataset_text_column="prompt", dataset_label_column="answer")
    preprocess = build_preprocess_fn(cfg, DummyTokenizer())
    with pytest.raises(KeyError):
        preprocess({"prompt": ["hello"]})


def test_build_training_args_v5_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training args should use eval_strategy and disable eval when absent."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(
        eval_strategy="steps",
        save_strategy="steps",
        seed=123,
        data_seed=321,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=1,
        max_target_length=16,
        max_label_count=2,
    )
    with_eval = build_training_args(cfg, has_eval=True)
    without_eval = build_training_args(cfg, has_eval=False)
    assert with_eval.eval_strategy.value == "steps"
    assert with_eval.eval_steps == cfg.eval_steps
    assert without_eval.eval_strategy.value == "no"
    assert without_eval.eval_steps is None
    assert with_eval.seed == 123
    assert with_eval.data_seed == 321
    assert with_eval.dataloader_num_workers == 2
    assert with_eval.dataloader_pin_memory is False
    assert with_eval.dataloader_persistent_workers is True
    assert with_eval.dataloader_prefetch_factor == 1
    assert with_eval.generation_max_length == cfg.resolved_max_target_length()


def test_build_training_args_defaults_data_seed_to_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data seed should fall back to seed when data_seed is not set."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(seed=77, data_seed=None)
    args = build_training_args(cfg, has_eval=True)
    assert args.seed == 77
    assert args.data_seed == 77


def test_build_training_args_disables_persistent_workers_without_worker_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent workers should be disabled when dataloader workers are zero."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
    )
    args = build_training_args(cfg, has_eval=True)
    assert args.dataloader_num_workers == 0
    assert args.dataloader_pin_memory is False
    assert args.dataloader_persistent_workers is False
    assert args.dataloader_prefetch_factor is None


def test_build_training_args_honors_explicit_generation_max_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit generation cap should override derived target max length."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(max_target_length=16, max_label_count=4)
    args = build_training_args(cfg, has_eval=True, generation_max_length=19)
    assert args.generation_max_length == 19


def test_build_training_args_honors_stage_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage overrides should control output, epochs, lr, and warmup."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(
        output_dir="/tmp/base-output",
        num_train_epochs=4,
        lr=1e-5,
        warmup_steps=100,
        lr_scheduler_type="linear",
    )
    args = build_training_args(
        cfg,
        has_eval=True,
        output_dir="/tmp/stage-output",
        num_train_epochs=2,
        learning_rate=3e-5,
        warmup_steps=0,
        lr_scheduler_type="constant",
    )
    assert args.output_dir == "/tmp/stage-output"
    assert args.num_train_epochs == 2
    assert args.learning_rate == pytest.approx(3e-5)
    assert args.warmup_steps == 0
    scheduler = (
        args.lr_scheduler_type.value
        if hasattr(args.lr_scheduler_type, "value")
        else str(args.lr_scheduler_type)
    )
    assert scheduler == "constant"


def test_build_training_args_disables_fp16_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit disable flag should force fp16 AMP off."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(torch_dtype="float16")
    args = build_training_args(cfg, has_eval=True, disable_fp16=True)
    assert args.fp16 is False


def test_scope_metric_logs_for_pretrain_stage() -> None:
    """Pretraining metrics should be logged under the pretraining category."""
    logs = {"loss": 1.2, "eval_loss": 0.9, "epoch": 1.0}
    scoped = trainer_logging_module.scope_metric_logs_for_stage(logs, "pretrain")

    assert scoped["pretraining/loss"] == 1.2
    assert scoped["pretraining/val/loss"] == 0.9
    assert scoped["epoch"] == 1.0
    assert "loss" not in scoped
    assert "eval_loss" not in scoped


def test_scope_metric_logs_for_pretraining_stage_alias() -> None:
    """Pretraining stage aliases should share the same metric namespace."""
    logs = {"loss": 1.2, "eval_loss": 0.9}
    scoped = trainer_logging_module.scope_metric_logs_for_stage(logs, "pretraining")

    assert scoped["pretraining/loss"] == 1.2
    assert scoped["pretraining/val/loss"] == 0.9


def test_rewrite_logs_preserving_scoped_metric_keys() -> None:
    """W&B log rewriting should keep already-scoped keys unchanged."""
    logs = {
        "loss": 1.2,
        "eval_loss": 0.9,
        "test_f1": 0.8,
        "pretraining/val/loss": 0.7,
    }
    rewritten = wandb_utils_module.rewrite_logs_preserving_scoped_metric_keys(logs)

    assert rewritten["train/loss"] == 1.2
    assert rewritten["eval/loss"] == 0.9
    assert rewritten["test/f1"] == 0.8
    assert rewritten["pretraining/val/loss"] == 0.7
    assert "train/pretraining/val/loss" not in rewritten


def test_patch_transformers_wandb_log_rewrite_preserves_scoped_keys() -> None:
    """Transformers rewrite hook should preserve stage-scoped metric keys."""
    from transformers.integrations import integration_utils

    original_rewrite = integration_utils.rewrite_logs
    try:
        wandb_utils_module.patch_transformers_wandb_log_rewrite(report_to=["wandb"])
        rewritten = integration_utils.rewrite_logs(
            {"loss": 1.2, "pretraining/val/loss": 0.7}
        )
        assert rewritten["train/loss"] == 1.2
        assert rewritten["pretraining/val/loss"] == 0.7
        assert "train/pretraining/val/loss" not in rewritten
    finally:
        integration_utils.rewrite_logs = original_rewrite


def test_scope_metric_logs_for_finetune_stage() -> None:
    """Fine-tuning metrics should route to train/val/test categories."""
    logs = {"loss": 1.2, "eval_accuracy": 0.8, "test_f1": 0.7}
    scoped = trainer_logging_module.scope_metric_logs_for_stage(logs, "finetune")

    assert scoped["train/loss"] == 1.2
    assert scoped["val/accuracy"] == 0.8
    assert scoped["test/f1"] == 0.7


def test_evaluate_every_n_epochs_callback_skips_non_interval_epoch(
    tmp_path: Path,
) -> None:
    """Eval callback should skip intermediate epochs outside the interval."""
    callback = trainer_logging_module.EvaluateEveryNEpochsCallback(every_n_epochs=10)
    args = trainer_logging_module.TrainingArguments(output_dir=str(tmp_path / "out"))
    state = trainer_logging_module.TrainerState()
    state.epoch = 9.0
    state.num_train_epochs = 20
    control = trainer_logging_module.TrainerControl(should_evaluate=True)

    updated = callback.on_epoch_end(args=args, state=state, control=control)

    assert updated.should_evaluate is False


def test_evaluate_every_n_epochs_callback_runs_on_interval_and_final_epoch(
    tmp_path: Path,
) -> None:
    """Eval callback should evaluate on interval epochs and final epoch."""
    callback = trainer_logging_module.EvaluateEveryNEpochsCallback(every_n_epochs=10)
    args = trainer_logging_module.TrainingArguments(output_dir=str(tmp_path / "out"))

    interval_state = trainer_logging_module.TrainerState()
    interval_state.epoch = 10.0
    interval_state.num_train_epochs = 20
    interval_control = trainer_logging_module.TrainerControl(should_evaluate=True)
    interval_updated = callback.on_epoch_end(
        args=args,
        state=interval_state,
        control=interval_control,
    )
    assert interval_updated.should_evaluate is True

    final_state = trainer_logging_module.TrainerState()
    final_state.epoch = 11.0
    final_state.num_train_epochs = 11
    final_control = trainer_logging_module.TrainerControl(should_evaluate=True)
    final_updated = callback.on_epoch_end(
        args=args,
        state=final_state,
        control=final_control,
    )
    assert final_updated.should_evaluate is True


def test_build_training_args_best_save_strategy_sets_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best save strategy should configure the best-model metric and direction."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(
        eval_strategy="epoch",
        save_strategy="best",
        save_strategy_best_metric="macro_f1",
    )
    args = build_training_args(cfg, has_eval=True)
    assert args.save_strategy.value == "best"
    assert args.metric_for_best_model == "macro_f1"
    assert args.greater_is_better is True


def test_build_training_args_best_loss_metric_uses_lower_is_better(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best save strategy should minimize loss metrics."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(save_strategy="best", save_strategy_best_metric="loss")
    args = build_training_args(cfg, has_eval=True)
    assert args.metric_for_best_model == "loss"
    assert args.greater_is_better is False


def test_build_training_args_best_save_strategy_requires_eval_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best save strategy should fail fast when no eval split is available."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(save_strategy="best")
    with pytest.raises(ValueError, match="requires validation data"):
        build_training_args(cfg, has_eval=False)


def test_build_training_args_best_micro_metric_requires_multi_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best save strategy should reject micro metrics for single-label runs."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(
        save_strategy="best",
        save_strategy_best_metric="micro_f1",
        max_label_count=1,
    )
    with pytest.raises(ValueError, match="max_label_count > 1"):
        build_training_args(cfg, has_eval=True)


def test_print_training_configuration_emits_summary_when_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbose mode should print JSON summary with config and training args."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(verbose=True, training_input=["cod", "age"], hf_token="secret")
    args = build_training_args(cfg, has_eval=True)
    train_module._print_training_configuration(
        cfg=cfg,
        args=args,
        run_data_metadata={"split_rows": {"train": 3}},
    )

    output = capsys.readouterr().out
    assert "Resolved training setup:" in output
    payload = json.loads(output.split("\n", maxsplit=1)[1])
    assert payload["config"]["training_input"] == ["cod", "age"]
    assert payload["config"]["verbose"] is True
    assert payload["training_args"]["learning_rate"] == cfg.lr
    assert payload["dataset"]["split_rows"]["train"] == 3
    assert "hf_token" not in payload["config"]


def test_print_training_configuration_noop_when_not_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-verbose mode should not print config summary."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(verbose=False)
    args = build_training_args(cfg, has_eval=True)
    train_module._print_training_configuration(
        cfg=cfg,
        args=args,
        run_data_metadata={"split_rows": {"train": 3}},
    )

    assert capsys.readouterr().out == ""


def test_exact_match_accuracy_is_one_for_identical_predictions() -> None:
    """Exact-match metric should be 1.0 when decoded predictions equal labels."""
    metric_fn = metrics_module.build_exact_match_accuracy_metric(DummyDecodeTokenizer())
    predictions = np.array([[1, 0, 0], [2, 0, 0]])
    labels = np.array([[1, -100, -100], [2, -100, -100]])
    metrics = metric_fn((predictions, labels))
    assert metrics["accuracy"] == 1.0
    assert "micro_precision" not in metrics


def test_exact_match_accuracy_reports_partial_match() -> None:
    """Exact-match metric should reflect prediction/label mismatches."""
    metric_fn = metrics_module.build_exact_match_accuracy_metric(DummyDecodeTokenizer())
    predictions = np.array([[1, 0, 0], [3, 0, 0]])
    labels = np.array([[1, -100, -100], [2, -100, -100]])
    metrics = metric_fn((predictions, labels))
    assert metrics["accuracy"] == 0.5
    assert "micro_precision" not in metrics


def test_exact_match_metric_reports_micro_for_multi_label() -> None:
    """Metric callback should expose micro metrics only in multi-label mode."""
    metric_fn = metrics_module.build_exact_match_accuracy_metric(
        DummyDecodeTokenizer(), label_separator=" ", max_label_count=2
    )
    predictions = np.array([[1, 2, 0], [3, 0, 0]])
    labels = np.array([[1, 2, -100], [2, -100, -100]])
    metrics = metric_fn((predictions, labels))
    assert metrics["accuracy"] == 0.5
    assert metrics["micro_precision"] == pytest.approx(2 / 3)
    assert metrics["micro_recall"] == pytest.approx(2 / 3)
    assert metrics["micro_f1"] == pytest.approx(2 / 3)


def test_build_training_args_auto_dtype_disables_fp16_without_bf16_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto dtype on CUDA should avoid fp16 when bf16 AMP is unavailable."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(train_module.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        train_module.torch.cuda, "is_bf16_supported", lambda: False, raising=False
    )
    cfg = Config(torch_dtype="auto")
    with pytest.warns(UserWarning, match="full precision"):
        args = build_training_args(cfg, has_eval=True)
    assert args.fp16 is False
    assert args.bf16 is False


def test_build_training_args_auto_dtype_uses_bf16_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto dtype on CUDA should prefer bf16 AMP when hardware supports it."""
    monkeypatch.setattr(
        train_module.wandb_utils,
        "resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(train_module.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        train_module.torch.cuda, "is_bf16_supported", lambda: True, raising=False
    )
    cfg = Config(torch_dtype="auto")
    args = build_training_args(cfg, has_eval=True)
    assert args.fp16 is False
    assert args.bf16 is True


def test_model_uses_trainable_fp16_params_detects_half_weights() -> None:
    """Model inspection should detect trainable float16 parameters."""
    model = train_module.torch.nn.Linear(4, 2).half()
    assert train_module._model_uses_trainable_fp16_params(model) is True


def test_model_uses_trainable_fp16_params_ignores_frozen_weights() -> None:
    """Frozen float16 parameters should not trigger fp16 AMP disablement."""
    model = train_module.torch.nn.Linear(4, 2).half()
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert train_module._model_uses_trainable_fp16_params(model) is False


def test_upcast_trainable_fp16_params_casts_to_float32() -> None:
    """Upcast helper should convert trainable half weights to float32."""
    model = train_module.torch.nn.Linear(4, 2).half()
    did_upcast = train_module._upcast_trainable_fp16_params(model)
    assert did_upcast is True
    dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.requires_grad
    }
    assert dtypes == {train_module.torch.float32}


def test_upcast_trainable_fp16_params_noop_for_float32() -> None:
    """Upcast helper should no-op when model is already float32."""
    model = train_module.torch.nn.Linear(4, 2).float()
    did_upcast = train_module._upcast_trainable_fp16_params(model)
    assert did_upcast is False


def test_validate_trainable_model_rejects_quantized_model() -> None:
    """Quantized base models should fail with an actionable error."""

    class QuantizedModel:
        """Model stub that mimics quantized loading."""

        is_quantized = True

    with pytest.raises(ValueError, match="CODLLM_LOAD_IN_8BIT"):
        train_module._validate_trainable_model(QuantizedModel())


def test_validate_trainable_model_accepts_non_quantized_model() -> None:
    """Non-quantized models should pass validation."""

    class TrainableModel:
        """Model stub that mimics regular dense loading."""

        is_quantized = False

    train_module._validate_trainable_model(TrainableModel())


def test_resolve_wandb_reporting_uses_wandb_with_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should enable W&B when credentials are available."""
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_LOG_MODEL", raising=False)
    monkeypatch.setattr(wandb_utils_module, "has_wandb_credentials", lambda: True)

    cfg = Config(
        wandb=WandbConfig(
            enabled=True,
            project="unit-project",
            entity="unit-entity",
            run_name="unit-run",
            mode="auto",
            log_model="checkpoint",
        )
    )
    report_to, run_name = wandb_utils_module.resolve_wandb_reporting(cfg)

    assert report_to == ["wandb"]
    assert run_name == "unit-run"
    assert wandb_utils_module.os.environ["WANDB_PROJECT"] == "unit-project"
    assert wandb_utils_module.os.environ["WANDB_ENTITY"] == "unit-entity"
    assert wandb_utils_module.os.environ["WANDB_LOG_MODEL"] == "checkpoint"


def test_resolve_wandb_reporting_disables_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should gracefully disable W&B when credentials are unavailable."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_LOG_MODEL", raising=False)
    monkeypatch.setattr(wandb_utils_module, "has_wandb_credentials", lambda: False)
    cfg = Config(wandb=WandbConfig(enabled=True, mode="auto"))

    with pytest.warns(UserWarning):
        report_to, run_name = wandb_utils_module.resolve_wandb_reporting(cfg)

    assert report_to == "none"
    assert run_name is None
    assert wandb_utils_module.os.environ["WANDB_MODE"] == "disabled"
    assert wandb_utils_module.os.environ["WANDB_LOG_MODEL"] == "false"


def test_resolve_wandb_reporting_online_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Online mode should always report to W&B and set WANDB_MODE accordingly."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_LOG_MODEL", raising=False)
    cfg = Config(wandb=WandbConfig(enabled=True, mode="online", run_name="online-run"))
    report_to, run_name = wandb_utils_module.resolve_wandb_reporting(cfg)

    assert report_to == ["wandb"]
    assert run_name == "online-run"
    assert wandb_utils_module.os.environ["WANDB_MODE"] == "online"
    assert wandb_utils_module.os.environ["WANDB_LOG_MODEL"] == "end"


def test_train_uses_validation_split_from_data_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train should forward train and validation splits to the training core."""
    cfg = Config(output_dir=str(tmp_path / "runs"))
    splits = DataSplits(
        train=pd.DataFrame({"text": ["t1", "t2"], "label": ["A00", "A01"]}),
        val=pd.DataFrame({"text": ["v1"], "label": ["A02"]}),
        test=pd.DataFrame({"text": ["e1"], "label": ["A03"]}),
    )

    class DummyDataHandler:
        """Stub data handler that returns fixed splits."""

        def __init__(self) -> None:
            self.force_reprocess: bool | None = None

        def get_splits(self, force_reprocess: bool = False) -> DataSplits:
            self.force_reprocess = force_reprocess
            return splits

    captured: dict[str, Any] = {}

    def fake_train(
        train_cfg: Config,
        train_ds: Any,
        eval_ds: Optional[Any] = None,
        run_data_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        captured["cfg"] = train_cfg
        captured["train_ds"] = train_ds
        captured["eval_ds"] = eval_ds
        captured["run_data_metadata"] = run_data_metadata
        return "trainer", "tokenizer"

    monkeypatch.setattr(train_module, "_train_from_datasets", fake_train)
    handler = DummyDataHandler()
    trainer, tokenizer, returned_splits = train_module.train(
        cfg, data_handler=handler, force_reprocess=True
    )

    assert trainer == "trainer"
    assert tokenizer == "tokenizer"
    assert returned_splits is splits
    assert captured["cfg"] is cfg
    assert captured["train_ds"] is splits.train
    assert captured["eval_ds"] is splits.val
    assert captured["run_data_metadata"]["split_rows"] == {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    assert handler.force_reprocess is True
    assert Path(captured["cfg"].output_dir).name.startswith("run-")
    assert Path(captured["cfg"].output_dir).parent.name == "runs"


def test_train_sequence_classification_builds_masterlist_label_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train should derive classifier label mappings from masterlist labels."""
    cfg = Config(
        output_dir=str(tmp_path / "runs"),
        model_task="sequence_classification",
        max_label_count=1,
    )
    splits = DataSplits(
        train=pd.DataFrame({"text": ["t1", "t2"], "label": ["A00", "A01"]}),
        val=pd.DataFrame({"text": ["v1"], "label": ["A00"]}),
        test=pd.DataFrame({"text": ["x1"], "label": ["A01"]}),
    )

    class DummyDataHandler:
        """Stub data handler returning fixed splits and masterlist labels."""

        def get_splits(self, force_reprocess: bool = False) -> DataSplits:
            return splits

        def get_masterlist_label_vocabulary(self) -> list[str]:
            return ["A00", "A01", "A02"]

    captured: dict[str, Any] = {}

    def fake_train(
        train_cfg: Config,
        train_ds: Any,
        eval_ds: Optional[Any] = None,
        run_data_metadata: Optional[dict[str, Any]] = None,
        label2id: Optional[dict[str, int]] = None,
        id2label: Optional[dict[int, str]] = None,
    ) -> tuple[str, str]:
        captured["cfg"] = train_cfg
        captured["train_ds"] = train_ds
        captured["eval_ds"] = eval_ds
        captured["run_data_metadata"] = run_data_metadata
        captured["label2id"] = label2id
        captured["id2label"] = id2label
        return "trainer", "tokenizer"

    captured_test_eval: dict[str, Any] = {}

    def fake_evaluate_test_split(
        cfg: Config,
        trainer: Any,
        tokenizer: Any,
        test_ds: Any,
        label2id: Optional[dict[str, int]] = None,
    ) -> dict[str, float]:
        captured_test_eval["cfg"] = cfg
        captured_test_eval["trainer"] = trainer
        captured_test_eval["tokenizer"] = tokenizer
        captured_test_eval["test_ds"] = test_ds
        captured_test_eval["label2id"] = label2id
        return {"test_accuracy": 1.0}

    monkeypatch.setattr(train_module, "_train_from_datasets", fake_train)
    monkeypatch.setattr(
        train_module,
        "_evaluate_test_split",
        fake_evaluate_test_split,
    )

    trainer, tokenizer, returned_splits = train_module.train(
        cfg, data_handler=DummyDataHandler()
    )

    assert trainer == "trainer"
    assert tokenizer == "tokenizer"
    assert returned_splits is splits
    assert captured["train_ds"] is splits.train
    assert captured["eval_ds"] is splits.val
    assert captured["label2id"] == {"A00": 0, "A01": 1, "A02": 2}
    assert captured["id2label"] == {0: "A00", 1: "A01", 2: "A02"}
    assert captured["run_data_metadata"]["classification"]["num_labels"] == 3
    assert captured_test_eval["test_ds"] is splits.test
    assert captured_test_eval["label2id"] == {"A00": 0, "A01": 1, "A02": 2}


def test_train_sequence_classification_requires_single_label_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sequence classification should reject multi-label training mode."""
    cfg = Config(
        output_dir=str(tmp_path / "runs"),
        model_task="sequence_classification",
        max_label_count=2,
    )
    splits = DataSplits(
        train=pd.DataFrame({"text": ["t1"], "label": ["A00"]}),
        val=pd.DataFrame({"text": ["v1"], "label": ["A01"]}),
        test=pd.DataFrame({"text": ["x1"], "label": ["A02"]}),
    )

    class DummyDataHandler:
        """Stub data handler returning fixed splits and masterlist labels."""

        def get_splits(self, force_reprocess: bool = False) -> DataSplits:
            return splits

        def get_masterlist_label_vocabulary(self) -> list[str]:
            return ["A00", "A01"]

    monkeypatch.setattr(
        train_module,
        "_train_from_datasets",
        lambda *args, **kwargs: ("trainer", "tokenizer"),
    )

    with pytest.raises(ValueError, match="max_label_count=1"):
        train_module.train(cfg, data_handler=DummyDataHandler())


def test_train_omits_eval_when_validation_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train should pass eval_ds=None when validation split is empty."""
    cfg = Config(output_dir=str(tmp_path / "runs"))
    splits = DataSplits(
        train=pd.DataFrame({"text": ["t1"], "label": ["A00"]}),
        val=pd.DataFrame(columns=["text", "label"]),
        test=pd.DataFrame({"text": ["e1"], "label": ["A01"]}),
    )

    class DummyDataHandler:
        """Stub data handler that returns fixed splits."""

        def get_splits(self, force_reprocess: bool = False) -> DataSplits:
            return splits

    captured: dict[str, Any] = {}

    def fake_train(
        train_cfg: Config,
        train_ds: Any,
        eval_ds: Optional[Any] = None,
        run_data_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        captured["cfg"] = train_cfg
        captured["train_ds"] = train_ds
        captured["eval_ds"] = eval_ds
        captured["run_data_metadata"] = run_data_metadata
        return "trainer", "tokenizer"

    monkeypatch.setattr(train_module, "_train_from_datasets", fake_train)
    _, _, _ = train_module.train(cfg, data_handler=DummyDataHandler())

    assert captured["cfg"] is cfg
    assert captured["train_ds"] is splits.train
    assert captured["eval_ds"] is None
    assert captured["run_data_metadata"]["split_rows"] == {
        "train": 1,
        "val": 0,
        "test": 1,
    }


def test_resolve_run_output_dir_uses_hpc_job_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run directory should use LSF job identifiers when present."""
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LSB_JOBINDEX", "2")

    run_dir = run_directory_module.resolve_run_output_dir(str(tmp_path / "runs"))

    assert run_dir.name == "run-12345_2"
    assert run_dir.exists()


def test_resolve_run_output_dir_local_allocates_next_numeric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local run directory should increment from existing run folders."""
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBINDEX", raising=False)
    runs_dir = tmp_path / "runs"
    (runs_dir / "run-0001").mkdir(parents=True, exist_ok=True)
    (runs_dir / "run-0003").mkdir(parents=True, exist_ok=True)

    run_dir = run_directory_module.resolve_run_output_dir(str(runs_dir))

    assert run_dir.name == "run-0004"
    assert run_dir.exists()


def test_train_runs_final_test_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train should run a final evaluation pass on the test split."""
    cfg = Config(output_dir=str(tmp_path / "runs"))
    splits = DataSplits(
        train=pd.DataFrame({"text": ["t1"], "label": ["A00"]}),
        val=pd.DataFrame({"text": ["v1"], "label": ["A01"]}),
        test=pd.DataFrame({"text": ["x1", "x2"], "label": ["A02", "A03"]}),
    )

    class DummyDataHandler:
        """Stub data handler that returns fixed splits."""

        def get_splits(self, force_reprocess: bool = False) -> DataSplits:
            return splits

    monkeypatch.setattr(
        train_module,
        "_train_from_datasets",
        lambda *args, **kwargs: ("trainer", "tokenizer"),
    )
    captured: dict[str, Any] = {}

    def fake_evaluate_test_split(
        cfg: Config, trainer: Any, tokenizer: Any, test_ds: Any
    ) -> dict[str, float]:
        captured["cfg"] = cfg
        captured["trainer"] = trainer
        captured["tokenizer"] = tokenizer
        captured["test_ds"] = test_ds
        return {"test_accuracy": 1.0}

    monkeypatch.setattr(train_module, "_evaluate_test_split", fake_evaluate_test_split)
    train_module.train(cfg, data_handler=DummyDataHandler())

    assert captured["cfg"] is cfg
    assert captured["trainer"] == "trainer"
    assert captured["tokenizer"] == "tokenizer"
    assert captured["test_ds"] is splits.test


def test_train_uses_pretraining_dataset_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train should route to two-stage training when pretraining data exists."""
    cfg = Config(
        output_dir=str(tmp_path / "runs"),
        pretrain_enabled=True,
        pretrain_masterlist_path="data/raw/ICD10h_Masterlist_2024.xlsx",
        pretrain_masterlist_sheet_name="Masterlist",
        pretrain_num_train_epochs=2,
        pretrain_learning_rate=7e-6,
        pretrain_eval_every_n_epochs=10,
        pretrain_lr_scheduler_type="constant",
    )
    splits = DataSplits(
        train=pd.DataFrame({"text": ["t1", "t2"], "label": ["A00", "A01"]}),
        val=pd.DataFrame({"text": ["v1"], "label": ["A02"]}),
        test=pd.DataFrame({"text": ["x1"], "label": ["A03"]}),
    )
    pretrain_df = pd.DataFrame({"text": ["p1", "p2"], "label": ["A10", "A11"]})

    class DummyDataHandler:
        """Stub data handler returning regular and pretraining dataframes."""

        def get_splits(self, force_reprocess: bool = False) -> DataSplits:
            return splits

        def get_pretraining_train_dataframe(self) -> pd.DataFrame:
            return pretrain_df

    captured: dict[str, Any] = {}

    def fake_train_with_optional_pretraining(
        cfg: Config,
        pretrain_ds: Any,
        train_ds: Any,
        eval_ds: Optional[Any] = None,
        run_data_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        captured["cfg"] = cfg
        captured["pretrain_ds"] = pretrain_ds
        captured["train_ds"] = train_ds
        captured["eval_ds"] = eval_ds
        captured["run_data_metadata"] = run_data_metadata
        return "trainer", "tokenizer"

    monkeypatch.setattr(
        train_module,
        "_train_with_pretraining",
        fake_train_with_optional_pretraining,
    )
    monkeypatch.setattr(train_module, "_evaluate_test_split", lambda **_: {"test": 1.0})

    trainer, tokenizer, returned_splits = train_module.train(
        cfg, data_handler=DummyDataHandler()
    )

    assert trainer == "trainer"
    assert tokenizer == "tokenizer"
    assert returned_splits is splits
    assert captured["pretrain_ds"] is pretrain_df
    assert captured["train_ds"] is splits.train
    assert captured["eval_ds"] is splits.val
    assert captured["run_data_metadata"]["pretraining"]["enabled"] is True
    assert captured["run_data_metadata"]["pretraining"]["train_rows"] == 2
    assert captured["run_data_metadata"]["pretraining"]["learning_rate"] == pytest.approx(7e-6)
    assert captured["run_data_metadata"]["pretraining"]["warmup_steps"] == 0
    assert captured["run_data_metadata"]["pretraining"]["eval_every_n_epochs"] == 10
    assert captured["run_data_metadata"]["pretraining"]["lr_scheduler_type"] == "constant"


def test_train_with_pretraining_uses_stage_specific_hyperparameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two-stage flow should pass pretrain-specific optimizer/eval settings."""
    cfg = Config(
        output_dir=str(tmp_path / "runs"),
        lr=2e-5,
        lr_scheduler_type="linear",
        warmup_steps=300,
        pretrain_num_train_epochs=5,
        pretrain_learning_rate=9e-6,
        pretrain_eval_every_n_epochs=10,
        pretrain_lr_scheduler_type="constant",
    )
    pretrain_ds = pd.DataFrame({"text": ["p1"], "label": ["A10"]})
    train_ds = pd.DataFrame({"text": ["t1"], "label": ["A00"]})
    eval_ds = pd.DataFrame({"text": ["v1"], "label": ["A01"]})

    monkeypatch.setattr(
        train_module,
        "_initialize_training_components",
        lambda _: ("model", "tokenizer", False),
    )
    captured_stages: list[dict[str, Any]] = []
    release_calls: list[str] = []

    class DummyTrainer:
        """Trainer stub that carries stage identity for assertions."""

        def __init__(self, stage_name: str) -> None:
            self.stage_name = stage_name

    def fake_train_with_model(
        cfg: Config,
        model: Any,
        tokenizer: Any,
        disable_fp16: bool,
        train_ds: Any,
        eval_ds: Optional[Any] = None,
        run_data_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        del cfg, model, tokenizer, disable_fp16, train_ds, eval_ds
        assert run_data_metadata is not None
        stage = run_data_metadata.get("training_stage")
        assert isinstance(stage, dict)
        captured_stages.append(stage)
        return DummyTrainer(stage_name=str(stage["name"]))

    monkeypatch.setattr(train_module, "_train_with_model", fake_train_with_model)
    monkeypatch.setattr(
        train_module,
        "_release_stage_trainer_memory",
        lambda cfg, trainer: release_calls.append(str(trainer.stage_name)),
    )

    trainer, tokenizer = train_module._train_with_pretraining(
        cfg=cfg,
        pretrain_ds=pretrain_ds,
        train_ds=train_ds,
        eval_ds=eval_ds,
        run_data_metadata={"split_rows": {"train": 1, "val": 1, "test": 1}},
    )

    assert trainer.stage_name == "finetune"
    assert tokenizer == "tokenizer"
    assert len(captured_stages) == 2
    assert release_calls == ["pretrain"]
    pretrain_stage = captured_stages[0]
    finetune_stage = captured_stages[1]
    assert pretrain_stage["name"] == "pretrain"
    assert pretrain_stage["learning_rate"] == pytest.approx(9e-6)
    assert pretrain_stage["warmup_steps"] == 0
    assert pretrain_stage["eval_every_n_epochs"] == 10
    assert pretrain_stage["lr_scheduler_type"] == "constant"
    assert Path(pretrain_stage["output_dir"]).name == "pretrain"
    assert finetune_stage["name"] == "finetune"
    assert finetune_stage["learning_rate"] == pytest.approx(2e-5)
    assert finetune_stage["warmup_steps"] == 300
    assert finetune_stage["lr_scheduler_type"] == "linear"
    assert Path(finetune_stage["output_dir"]).name == "finetune"


def test_release_stage_trainer_memory_clears_state_and_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory release helper should clear trainer state and flush CUDA cache."""
    cfg = Config()
    monkeypatch.setattr(cfg, "uses_cuda", lambda: True)
    state = {"gc_called": False, "empty_cache_called": False}
    monkeypatch.setattr(
        train_module.gc,
        "collect",
        lambda: state.__setitem__("gc_called", True),
    )
    monkeypatch.setattr(
        train_module.torch.cuda,
        "empty_cache",
        lambda: state.__setitem__("empty_cache_called", True),
    )

    class DummyTrainer:
        """Trainer stub exposing optimizer and scheduler attributes."""

        optimizer: Any = object()
        lr_scheduler: Any = object()

    trainer = DummyTrainer()
    train_module._release_stage_trainer_memory(cfg=cfg, trainer=trainer)

    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert state["gc_called"] is True
    assert state["empty_cache_called"] is True


def test_train_pretraining_requires_validation_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pretraining should fail fast when regular validation split is empty."""
    cfg = Config(
        output_dir=str(tmp_path / "runs"),
        pretrain_enabled=True,
    )
    splits = DataSplits(
        train=pd.DataFrame({"text": ["t1"], "label": ["A00"]}),
        val=pd.DataFrame(columns=["text", "label"]),
        test=pd.DataFrame({"text": ["x1"], "label": ["A01"]}),
    )
    pretrain_df = pd.DataFrame({"text": ["p1"], "label": ["A10"]})

    class DummyDataHandler:
        """Stub data handler returning empty validation and pretraining rows."""

        def get_splits(self, force_reprocess: bool = False) -> DataSplits:
            return splits

        def get_pretraining_train_dataframe(self) -> pd.DataFrame:
            return pretrain_df

    monkeypatch.setattr(
        train_module,
        "_train_with_pretraining",
        lambda *args, **kwargs: ("trainer", "tokenizer"),
    )

    with pytest.raises(ValueError, match="requires a non-empty validation split"):
        train_module.train(cfg, data_handler=DummyDataHandler())


def test_evaluate_test_split_uses_test_metric_prefix() -> None:
    """Final test evaluation should call Trainer.evaluate with test key prefix."""
    cfg = Config()
    test_ds = pd.DataFrame({"text": ["x1", "x2"], "label": ["A00", "A01"]})

    class DummyTrainer:
        """Trainer stub that captures evaluate calls."""

        def __init__(self) -> None:
            self.called_with: dict[str, Any] = {}

        def evaluate(
            self, eval_dataset: Any, metric_key_prefix: str
        ) -> dict[str, float]:
            self.called_with["length"] = len(eval_dataset)
            self.called_with["metric_key_prefix"] = metric_key_prefix
            return {"test_accuracy": 1.0, "test_f1": 1.0}

    trainer = DummyTrainer()
    metrics = train_module._evaluate_test_split(cfg, trainer, DummyTokenizer(), test_ds)

    assert metrics == {"test_accuracy": 1.0, "test_f1": 1.0}
    assert trainer.called_with["metric_key_prefix"] == "test"
    assert trainer.called_with["length"] == 2


def test_evaluate_test_split_sequence_classification_uses_label_mapping() -> None:
    """Classification test evaluation should encode labels via label2id mapping."""
    cfg = Config(model_task="sequence_classification")
    test_ds = pd.DataFrame({"text": ["x1", "x2"], "label": ["A00", "A01"]})

    class DummyTrainer:
        """Trainer stub that captures evaluate calls."""

        def __init__(self) -> None:
            self.called_with: dict[str, Any] = {}

        def evaluate(
            self, eval_dataset: Any, metric_key_prefix: str
        ) -> dict[str, float]:
            self.called_with["length"] = len(eval_dataset)
            self.called_with["metric_key_prefix"] = metric_key_prefix
            self.called_with["labels"] = [
                eval_dataset[idx]["labels"] for idx in range(len(eval_dataset))
            ]
            return {"test_accuracy": 1.0}

    trainer = DummyTrainer()
    metrics = train_module._evaluate_test_split(
        cfg=cfg,
        trainer=trainer,
        tokenizer=DummyTokenizer(),
        test_ds=test_ds,
        label2id={"A00": 0, "A01": 1},
    )

    assert metrics == {"test_accuracy": 1.0}
    assert trainer.called_with["metric_key_prefix"] == "test"
    assert trainer.called_with["length"] == 2
    assert trainer.called_with["labels"] == [0, 1]


class _FakeWandbConfig:
    """Minimal W&B config stub that records updates."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, payload: dict[str, Any], allow_val_change: bool = False) -> None:
        self.updates.append(
            {
                "payload": payload,
                "allow_val_change": allow_val_change,
            }
        )


class _FakeWandbModule:
    """Minimal W&B module stub for metadata logging tests."""

    def __init__(self) -> None:
        self.run: object | None = None
        self.config = _FakeWandbConfig()
        self.init_calls: list[dict[str, Any]] = []

    def init(self, **kwargs: Any) -> object:
        self.init_calls.append(kwargs)
        self.run = object()
        return self.run


def test_build_experiment_metadata_redacts_hf_token() -> None:
    """Experiment metadata should omit sensitive Hugging Face token values."""
    cfg = Config(hf_token="super-secret-token", data_seed=None, seed=42)
    payload = wandb_utils_module.build_experiment_metadata(
        cfg, data_metadata={"split_rows": {"train": 10}}
    )

    assert payload["config"]["hf_model"] == cfg.hf_model
    assert "hf_token" not in payload["config"]
    assert payload["resolved"]["data_seed"] == 42
    assert payload["dataset"]["split_rows"]["train"] == 10


def test_build_experiment_metadata_includes_training_args() -> None:
    """Experiment metadata should include serialized Trainer argument values."""
    cfg = Config(seed=42)
    payload = wandb_utils_module.build_experiment_metadata(
        cfg,
        training_args={"learning_rate": 3e-5, "num_train_epochs": 4},
    )

    assert payload["training_args"]["learning_rate"] == 3e-5
    assert payload["training_args"]["num_train_epochs"] == 4


def test_log_wandb_run_metadata_initializes_and_updates_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata logger should initialize W&B and update run config once per call."""
    fake_wandb = _FakeWandbModule()
    monkeypatch.setattr(wandb_utils_module, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_MODE", "offline")

    cfg = Config(
        wandb=WandbConfig(
            enabled=True,
            project="cfg-project",
            entity="cfg-entity",
            run_name="cfg-run",
            mode="offline",
        )
    )
    metadata = {"dataset": {"split_rows": {"train": 3}}}
    wandb_utils_module.log_wandb_run_metadata(
        cfg=cfg,
        report_to=["wandb"],
        run_name="explicit-run",
        metadata=metadata,
    )

    assert len(fake_wandb.init_calls) == 1
    assert fake_wandb.init_calls[0]["project"] == "test-project"
    assert fake_wandb.init_calls[0]["entity"] == "cfg-entity"
    assert fake_wandb.init_calls[0]["name"] == "explicit-run"
    assert (
        fake_wandb.config.updates[0]["payload"]["dataset"]["split_rows"]["train"] == 3
    )
    assert fake_wandb.config.updates[0]["allow_val_change"] is True


def test_log_wandb_run_metadata_writes_flattened_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata logger should publish flattened dot-notation keys for filtering."""
    fake_wandb = _FakeWandbModule()
    monkeypatch.setattr(wandb_utils_module, "_import_wandb", lambda: fake_wandb)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_MODE", "offline")

    cfg = Config(
        wandb=WandbConfig(
            enabled=True,
            project="cfg-project",
            entity="cfg-entity",
            run_name="cfg-run",
            mode="offline",
        )
    )
    metadata = {
        "config": {"lr": 3e-5, "training_input": ["cod", "age"]},
        "dataset": {"split_rows": {"train": 3}},
    }
    wandb_utils_module.log_wandb_run_metadata(
        cfg=cfg,
        report_to=["wandb"],
        run_name="flattened-run",
        metadata=metadata,
    )

    flattened_updates = [
        update["payload"]
        for update in fake_wandb.config.updates
        if "config.lr" in update["payload"]
    ]
    assert flattened_updates
    assert flattened_updates[0]["config.lr"] == 3e-5
    assert flattened_updates[0]["config.training_input"] == ["cod", "age"]
    assert flattened_updates[0]["dataset.split_rows.train"] == 3


def test_log_wandb_run_metadata_noop_when_wandb_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata logger should no-op when training args disable W&B reporting."""
    fake_wandb = _FakeWandbModule()
    monkeypatch.setattr(wandb_utils_module, "_import_wandb", lambda: fake_wandb)

    cfg = Config(wandb=WandbConfig(enabled=False, mode="disabled"))
    wandb_utils_module.log_wandb_run_metadata(
        cfg=cfg,
        report_to="none",
        run_name=None,
        metadata={"dataset": {"split_rows": {"train": 5}}},
    )

    assert fake_wandb.init_calls == []
    assert fake_wandb.config.updates == []
