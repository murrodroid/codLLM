from __future__ import annotations

import argparse
import json
from pathlib import Path

from codllm.config import config_from_env
from codllm.inference.io import RECORD_ID_COLUMN
from codllm.inference.pipeline import InferenceRequest, run_inference


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the inference entrypoint."""
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a .txt, .csv, .tsv, .jsonl, or .parquet inference input file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional destination path for .csv, .tsv, .jsonl, or .parquet predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed before loading the model.",
    )
    parser.add_argument(
        "--validate-registry",
        action="store_true",
        help="Reject ICD10h codes that are absent from the configured masterlist registry.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Override the configured Hugging Face model identifier or local model path.",
    )
    parser.add_argument(
        "--model-task",
        choices=["seq2seq", "sequence_classification"],
        default=None,
        help="Override the configured inference task type.",
    )
    return parser.parse_args()


def _emit_stdout_predictions(outputs: object) -> None:
    """Emit compact JSONL predictions to stdout when no output file is requested."""
    if not hasattr(outputs, "iterrows"):
        raise TypeError("Inference outputs must be a pandas dataframe.")
    for _, row in outputs.iterrows():
        payload = {
            RECORD_ID_COLUMN: row[RECORD_ID_COLUMN],
            "prediction_label": row["prediction_label"],
            "prediction_raw_text": row["prediction_raw_text"],
            "prediction_invalid_codes_json": row["prediction_invalid_codes_json"],
        }
        print(json.dumps(payload, ensure_ascii=True))


def main() -> None:
    """Run the codllm inference CLI."""
    args = parse_args()
    cfg = config_from_env()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.hf_model is not None:
        cfg.hf_model = args.hf_model
    if args.model_task is not None:
        cfg.model_task = args.model_task

    outputs = run_inference(
        cfg=cfg,
        request=InferenceRequest(
            input_path=args.input_path,
            output_path=args.output_path,
            validate_registry=args.validate_registry,
        ),
    )
    if args.output_path is None:
        _emit_stdout_predictions(outputs)
        return
    print(f"Inference completed. rows={len(outputs)} output='{args.output_path}'.")
