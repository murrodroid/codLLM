from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from codllm.config import Config, WandbConfig
from codllm.data_handler import DataSplits
from codllm.preprocess import build_preprocess_fn
import codllm.train as train_module
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
        train_module,
        "_resolve_wandb_reporting",
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
    assert with_eval.generation_max_length == 21


def test_build_training_args_defaults_data_seed_to_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data seed should fall back to seed when data_seed is not set."""
    monkeypatch.setattr(
        train_module,
        "_resolve_wandb_reporting",
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
        train_module,
        "_resolve_wandb_reporting",
        lambda _: ("none", None),
    )
    cfg = Config(max_target_length=16, max_label_count=4)
    args = build_training_args(cfg, has_eval=True, generation_max_length=19)
    assert args.generation_max_length == 19


def test_resolve_wandb_reporting_uses_wandb_with_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should enable W&B when credentials are available."""
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(train_module, "_has_wandb_credentials", lambda: True)

    cfg = Config(
        wandb=WandbConfig(
            enabled=True,
            project="unit-project",
            entity="unit-entity",
            run_name="unit-run",
            mode="auto",
        )
    )
    report_to, run_name = train_module._resolve_wandb_reporting(cfg)

    assert report_to == ["wandb"]
    assert run_name == "unit-run"
    assert train_module.os.environ["WANDB_PROJECT"] == "unit-project"
    assert train_module.os.environ["WANDB_ENTITY"] == "unit-entity"


def test_resolve_wandb_reporting_disables_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should gracefully disable W&B when credentials are unavailable."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(train_module, "_has_wandb_credentials", lambda: False)
    cfg = Config(wandb=WandbConfig(enabled=True, mode="auto"))

    with pytest.warns(UserWarning):
        report_to, run_name = train_module._resolve_wandb_reporting(cfg)

    assert report_to == "none"
    assert run_name is None
    assert train_module.os.environ["WANDB_MODE"] == "disabled"


def test_resolve_wandb_reporting_online_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Online mode should always report to W&B and set WANDB_MODE accordingly."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    cfg = Config(wandb=WandbConfig(enabled=True, mode="online", run_name="online-run"))
    report_to, run_name = train_module._resolve_wandb_reporting(cfg)

    assert report_to == ["wandb"]
    assert run_name == "online-run"
    assert train_module.os.environ["WANDB_MODE"] == "online"


def test_train_with_data_handler_uses_validation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_with_data_handler should forward train and validation splits to train."""
    cfg = Config()
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
        train_cfg: Config, train_ds: Any, eval_ds: Optional[Any] = None
    ) -> tuple[str, str]:
        captured["cfg"] = train_cfg
        captured["train_ds"] = train_ds
        captured["eval_ds"] = eval_ds
        return "trainer", "tokenizer"

    monkeypatch.setattr(train_module, "train", fake_train)
    handler = DummyDataHandler()
    trainer, tokenizer, returned_splits = train_module.train_with_data_handler(
        cfg, data_handler=handler, force_reprocess=True
    )

    assert trainer == "trainer"
    assert tokenizer == "tokenizer"
    assert returned_splits is splits
    assert captured["cfg"] is cfg
    assert captured["train_ds"] is splits.train
    assert captured["eval_ds"] is splits.val
    assert handler.force_reprocess is True


def test_train_with_data_handler_omits_eval_when_validation_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_with_data_handler should pass eval_ds=None when validation split is empty."""
    cfg = Config()
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
        train_cfg: Config, train_ds: Any, eval_ds: Optional[Any] = None
    ) -> tuple[str, str]:
        captured["cfg"] = train_cfg
        captured["train_ds"] = train_ds
        captured["eval_ds"] = eval_ds
        return "trainer", "tokenizer"

    monkeypatch.setattr(train_module, "train", fake_train)
    _, _, _ = train_module.train_with_data_handler(cfg, data_handler=DummyDataHandler())

    assert captured["cfg"] is cfg
    assert captured["train_ds"] is splits.train
    assert captured["eval_ds"] is None
