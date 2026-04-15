from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest
import torch

from codllm.config import Config
from codllm.inference.io import RECORD_ID_COLUMN, load_inference_inputs
from codllm.inference.pipeline import InferenceRequest, run_inference
import codllm.inference.pipeline as pipeline_module


class DummyTokenizer:
    """Tokenizer stub for inference unit tests."""

    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        max_length: int,
        truncation: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del max_length, truncation, padding, return_tensors
        batch_size = len(texts)
        return {
            "input_ids": torch.arange(1, batch_size + 1, dtype=torch.int64).unsqueeze(
                1
            ),
            "attention_mask": torch.ones((batch_size, 1), dtype=torch.int64),
        }

    def batch_decode(
        self,
        token_ids: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        del skip_special_tokens
        mapping = {
            101: "A00.000",
            102: "B00.000",
            103: "BAD",
        }
        return [mapping[row[0]] for row in token_ids]


class DummySeq2SeqModel:
    """Seq2seq model stub for inference pipeline tests."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.last_max_new_tokens: int | None = None

    def eval(self) -> None:
        """No-op eval for compatibility."""

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        del attention_mask
        self.last_max_new_tokens = max_new_tokens
        tokens = [101 if int(value[0]) == 1 else 102 for value in input_ids]
        return torch.tensor([[token] for token in tokens], dtype=torch.int64)


class DummyClassifierModel:
    """Sequence-classification model stub for inference tests."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={0: "A00.000", 1: "B00.000"})

    def eval(self) -> None:
        """No-op eval for compatibility."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        return SimpleNamespace(
            logits=torch.tensor(
                [
                    [4.0, 1.0],
                    [1.0, 6.0],
                ],
                dtype=torch.float32,
            )
        )


def test_load_inference_inputs_reads_txt_lines(tmp_path: Path) -> None:
    """Plain-text inference inputs should become canonical dataframe rows."""
    input_path = tmp_path / "examples.txt"
    input_path.write_text("cholera\ntyphus\n")

    result = load_inference_inputs(input_path, text_column="text")

    assert list(result.columns) == [RECORD_ID_COLUMN, "text"]
    assert result.iloc[0][RECORD_ID_COLUMN] == "row-000000"
    assert result.iloc[1]["text"] == "typhus"


def test_run_inference_writes_seq2seq_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seq2seq inference should write parsed prediction columns to disk."""
    input_path = tmp_path / "inputs.csv"
    output_path = tmp_path / "predictions.csv"
    pd.DataFrame({"text": ["cholera", "typhus"]}).to_csv(input_path, index=False)

    model = DummySeq2SeqModel()
    tokenizer = DummyTokenizer()
    monkeypatch.setattr(
        pipeline_module,
        "load_base_model",
        lambda cfg: (model, tokenizer),
    )

    cfg = Config(
        per_device_eval_batch_size=2,
        max_target_length=8,
        max_label_count=1,
        label_separator=" | ",
    )
    outputs = run_inference(
        cfg=cfg,
        request=InferenceRequest(input_path=input_path, output_path=output_path),
    )

    written = pd.read_csv(output_path)
    assert model.last_max_new_tokens == cfg.resolved_max_target_length()
    assert outputs["prediction_label"].tolist() == ["A00.000", "B00.000"]
    assert written["prediction_invalid_code_count"].tolist() == [0, 0]


def test_run_inference_supports_sequence_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classifier inference should decode labels from model config id2label."""
    input_path = tmp_path / "inputs.csv"
    pd.DataFrame({"text": ["cholera", "typhus"]}).to_csv(input_path, index=False)

    monkeypatch.setattr(
        pipeline_module,
        "load_base_model",
        lambda cfg: (DummyClassifierModel(), DummyTokenizer()),
    )

    cfg = Config(
        model_task="sequence_classification",
        per_device_eval_batch_size=2,
        label_separator=" | ",
    )
    outputs = run_inference(
        cfg=cfg,
        request=InferenceRequest(input_path=input_path),
    )

    assert outputs["prediction_label"].tolist() == ["A00.000", "B00.000"]
    assert outputs["prediction_code_count"].tolist() == [1, 1]
