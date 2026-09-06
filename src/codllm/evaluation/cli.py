"""Frozen checkpoint evaluation without retraining or reconstructing the training split."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from codllm.config import Config, config_from_env
from codllm.evaluation.artifacts import PublicationReporter, load_manifest, write_json
from codllm.evaluation.context import evaluation_rows
from codllm.evaluation.provenance import evaluation_records, load_external_table
from codllm.evaluation.reference import EvaluationReference
from codllm.evaluation.workflow import validate_publication_config
from codllm.metrics import reset_metric_artifact_scope, set_metric_artifact_scope


def load_reference(directory: Path) -> EvaluationReference:
    """Load the adaptation inventory frozen by the original training run."""
    payload = json.loads((directory / "reference.json").read_text())["reference"]
    for name in (
        "historical_cods",
        "adapted_cods",
        "masterlist_cods",
        "combinations",
        "masterlist_labels",
    ):
        payload[name] = set(payload[name])
    for name in ("label_sources", "label_languages", "adaptation_languages"):
        payload[name] = {key: set(values) for key, values in payload[name].items()}
    return EvaluationReference(**payload)


def validate_evaluation_inputs(cfg: Config) -> Path:
    """Check explicit checkpoint/reference inputs before submission or inference."""
    if (
        cfg.evaluation_data_path or cfg.evaluation_scope == "test"
    ) and cfg.publication_gate != "final":
        raise ValueError(
            "External/test checkpoint evaluation requires a reviewed final gate."
        )
    if not cfg.evaluation_checkpoint or not Path(cfg.evaluation_checkpoint).is_dir():
        raise ValueError(
            "Set CODLLM_EVALUATION_CHECKPOINT to the frozen local checkpoint directory."
        )
    if not cfg.evaluation_reference_dir:
        raise ValueError(
            "Set CODLLM_EVALUATION_REFERENCE_DIR to the original run's publication directory."
        )
    directory = Path(cfg.evaluation_reference_dir)
    for name in ("reference.json", "contract.json", "effective_config.json"):
        if not (directory / name).is_file():
            raise ValueError(f"Missing frozen training artifact: {directory / name}")
    if cfg.evaluation_data_path and not Path(cfg.evaluation_data_path).is_file():
        raise ValueError("CODLLM_EVALUATION_DATA_PATH does not exist.")
    if not cfg.evaluation_data_path and cfg.evaluation_scope not in {
        "val",
        "test",
        "holdout",
    }:
        raise ValueError("External evaluation requires CODLLM_EVALUATION_DATA_PATH.")
    return directory


def checkpoint_digest(directory: Path) -> str:
    """Hash model weights and tokenizer/configuration files in a frozen checkpoint."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (
            path.suffix in {".json", ".safetensors", ".model"}
            or path.name == "pytorch_model.bin"
        )
    )
    if not any(
        path.suffix == ".safetensors" or path.name == "pytorch_model.bin"
        for path in files
    ):
        raise ValueError("Checkpoint contains no supported local model weights.")
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def evaluate_checkpoint(cfg: Config) -> dict[str, float]:
    """Evaluate a frozen model on a saved split or explicitly supplied external table."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    validate_publication_config(cfg)
    directory = validate_evaluation_inputs(cfg)
    frozen = json.loads((directory / "effective_config.json").read_text())
    for field in (
        "training_input",
        "input_field_prefixes",
        "text_field_separator",
        "dataset_text_column",
        "dataset_label_column",
        "max_source_length",
        "max_target_length",
        "max_label_count",
        "label_separator",
        "label_code_length",
        "max_target_length_buffer",
        "per_device_eval_batch_size",
        "torch_dtype",
    ):
        setattr(cfg, field, frozen[field])
    if frozen["model_task"] != "seq2seq":
        raise ValueError(
            "Frozen-checkpoint CLI currently supports seq2seq models only."
        )
    if cfg.evaluation_data_path:
        frame = load_external_table(cfg)
    else:
        frame = load_manifest(directory, cfg.evaluation_scope)
    if frame.empty:
        raise ValueError("Evaluation table is empty.")
    if frame["row_uid"].duplicated().any():
        raise ValueError("Evaluation table has duplicate row identities.")
    model_path = Path(cfg.evaluation_checkpoint)
    digest = checkpoint_digest(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path), local_files_only=True, dtype="auto"
        )
        .to(cfg.device)
        .eval()
    )
    decoded = []
    texts = frame[cfg.dataset_text_column].fillna("").astype(str).tolist()
    amp_dtype = None
    if cfg.uses_cuda():
        if cfg.torch_dtype == "bfloat16" or (
            cfg.torch_dtype == "auto" and cfg.bf16_amp_supported()
        ):
            amp_dtype = torch.bfloat16
        elif cfg.torch_dtype == "float16":
            amp_dtype = torch.float16
    precision = (
        torch.autocast("cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )
    with torch.inference_mode(), precision:
        for start in range(0, len(texts), cfg.per_device_eval_batch_size):
            inputs = tokenizer(
                texts[start : start + cfg.per_device_eval_batch_size],
                padding=True,
                truncation=True,
                max_length=cfg.max_source_length,
                return_tensors="pt",
            ).to(cfg.device)
            generated = model.generate(
                **inputs, max_length=cfg.resolved_max_target_length()
            )
            decoded.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    predictions = [
        {part.strip() for part in value.split(cfg.label_separator) if part.strip()}
        for value in decoded
    ]
    labels = [
        {part.strip() for part in str(value).split(cfg.label_separator) if part.strip()}
        for value in frame[cfg.dataset_label_column]
    ]
    reporter = PublicationReporter(cfg, load_reference(directory))
    reporter.exporting = True
    reporter.trainer = SimpleNamespace(
        state=SimpleNamespace(
            best_model_checkpoint=str(model_path),
            global_step=None,
            epoch=None,
            best_metric=None,
        )
    )
    rows_token = evaluation_rows.set(evaluation_records(frame, cfg))
    scope_token = set_metric_artifact_scope(cfg.evaluation_scope)
    try:
        metrics = reporter(predictions, labels)
    finally:
        evaluation_rows.reset(rows_token)
        reset_metric_artifact_scope(scope_token)
    write_json(
        Path(cfg.output_dir) / "publication" / "frozen_evaluation.json",
        {
            "checkpoint_sha256": digest,
            "reference_dir": str(directory),
            "scope": cfg.evaluation_scope,
            "evaluation_batch_size": cfg.per_device_eval_batch_size,
            "autocast_dtype": str(amp_dtype),
            "metrics": metrics,
        },
    )
    if cfg.uncertainty_eval:
        from codllm.uncertainty.end_of_training import run_end_of_training_uncertainty

        validation = load_manifest(directory, "val")
        run_end_of_training_uncertainty(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            test_df=frame,
            val_df=validation,
            run_dir=Path(cfg.output_dir),
        )
    from codllm.evaluation.telemetry import log_completed_evaluation

    log_completed_evaluation(
        cfg, {cfg.evaluation_scope: metrics}, {"checkpoint_sha256": digest}
    )
    return metrics


def main() -> None:
    """Run the environment-configured frozen checkpoint evaluation entrypoint."""
    metrics = evaluate_checkpoint(config_from_env())
    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if key
                in {
                    "pub_v1_accuracy",
                    "pub_v1_macro_f1_ref",
                    "pub_v1_source_mean_macro_f1_ref",
                }
            },
            indent=2,
        )
    )
