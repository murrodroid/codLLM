from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from codllm.config import Config
from codllm.models import load_base_model
from codllm.runtime.paths import resolve_source_path
from codllm.inference.decoding import (
    build_prediction_dataframe,
    configure_registry_validation,
)
from codllm.inference.generation import generate_predictions
from codllm.inference.io import load_inference_inputs, write_inference_outputs


@dataclass(frozen=True)
class InferenceRequest:
    """Filesystem request for one inference run."""

    input_path: Path
    output_path: Path | None = None
    validate_registry: bool = False


def run_inference(cfg: Config, request: InferenceRequest) -> pd.DataFrame:
    """Run end-to-end inference for one input file."""
    validate_registry = request.validate_registry or cfg.inference_validate_registry
    inputs = load_inference_inputs(
        path=request.input_path,
        text_column=cfg.dataset_text_column,
    )
    if validate_registry:
        registry_path = resolve_source_path(
            cfg.pretrain_masterlist_path, cfg.data_raw_dir
        )
        configure_registry_validation(str(registry_path))

    model, tokenizer = load_base_model(cfg)
    raw_predictions = generate_predictions(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        texts=inputs[cfg.dataset_text_column].tolist(),
    )
    outputs = build_prediction_dataframe(
        inputs=inputs,
        raw_predictions=raw_predictions,
        label_separator=cfg.label_separator,
        validate_registry=validate_registry,
    )
    if request.output_path is not None:
        write_inference_outputs(outputs, request.output_path)
    return outputs
