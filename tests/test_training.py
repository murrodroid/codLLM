import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

import codllm.metrics as metrics_module
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

    run_dir = train_module._resolve_run_output_dir(str(tmp_path / "runs"))

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

    run_dir = train_module._resolve_run_output_dir(str(runs_dir))

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
