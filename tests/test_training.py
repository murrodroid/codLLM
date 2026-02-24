from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from codllm.config import Config
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


def test_build_training_args_v5_compatible() -> None:
    """Training args should use eval_strategy and disable eval when absent."""
    cfg = Config(eval_strategy="steps", save_strategy="steps")
    with_eval = build_training_args(cfg, has_eval=True)
    without_eval = build_training_args(cfg, has_eval=False)
    assert with_eval.eval_strategy.value == "steps"
    assert with_eval.eval_steps == cfg.eval_steps
    assert without_eval.eval_strategy.value == "no"
    assert without_eval.eval_steps is None


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
