"""Explicit, copy-only continuation of completed publication runs into new identities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock

from codllm.config import Config
from codllm.evaluation.artifacts import training_recipe, write_json
from codllm.evaluation.provenance import content_digest


def storage_path(value: str) -> Path:
    """Resolve a continuation storage path without guessing a laptop/HPC storage root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        storage = os.environ.get("RUN_STORAGE_DIR")
        if not storage:
            raise ValueError(
                "Relative continuation paths require RUN_STORAGE_DIR; source hpc/env.sh."
            )
        path = Path(storage) / path
    return path.resolve()


def configuration_payload(cfg: Config) -> dict[str, Any]:
    """Serialize configuration without credentials, matching publication contracts."""
    payload = asdict(cfg)
    payload.pop("hf_token", None)
    payload["device"] = str(cfg.device)
    payload["device_map"] = str(cfg.device_map)
    return payload


def _json(path: Path) -> dict[str, Any]:
    """Read a required JSON object with a useful missing-file error."""
    if not path.is_file():
        raise ValueError(f"Required continuation input is missing: {path}")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    """Hash one file without loading model or optimizer tensors into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Require full training state, so missing optimizer/RNG files cannot silently reset training."""
    required = (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    )
    if any(
        not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
        for name in required
    ):
        raise ValueError(f"Incomplete resumable checkpoint: {checkpoint}")
    if any(p.is_symlink() or not p.is_file() for p in checkpoint.iterdir()):
        raise ValueError(
            "Checkpoint must contain ordinary files, not symlinks or subdirectories."
        )
    state = _json(checkpoint / "trainer_state.json")
    if checkpoint.name != f"checkpoint-{state['global_step']}":
        raise ValueError("Checkpoint step and directory name disagree.")
    return state


def inspect_parent(cfg: Config) -> dict[str, Any]:
    """Fail closed unless the named parent completed and only stopping settings changed."""
    if not all(
        (
            cfg.continuation_source_run_dir,
            cfg.continuation_source_state_dir,
            cfg.continuation_parent_wandb_run,
        )
    ):
        raise ValueError(
            "Continuation requires explicit parent run, state directory, and W&B path."
        )
    source = storage_path(cfg.continuation_source_run_dir)
    destination = storage_path(cfg.output_dir)
    state_dir = Path(cfg.continuation_source_state_dir).expanduser().resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError(
            "Continuation destination must be separate from the parent run."
        )
    if not destination.name.startswith("run-"):
        raise ValueError("Continuation output must name an explicit run-* directory.")
    if not all(
        (state_dir / name).is_file()
        for name in (".training_complete", ".final_eval_done")
    ):
        raise ValueError(f"Parent is not completed; wait before preparing: {source}")
    if (state_dir / ".resume_needed").exists():
        raise ValueError(f"Parent still requests resumption: {state_dir}")
    parent_id = (state_dir / "wandb_run_id.txt").read_text().strip()
    if (
        cfg.continuation_parent_wandb_run
        != f"{cfg.wandb.entity}/{cfg.wandb.project}/{parent_id}"
    ):
        raise ValueError(
            "Parent W&B identity does not match the explicit parent state directory."
        )
    parent_cfg = _json(source / "publication/effective_config.json")
    contract = _json(source / "publication/contract.json")
    old_recipe = training_recipe(parent_cfg)
    new_recipe = training_recipe(configuration_payload(cfg))
    if content_digest(old_recipe) != contract.get("training_recipe_digest"):
        raise ValueError(
            "Parent configuration does not match its frozen recipe digest."
        )
    differences = {
        key
        for key in old_recipe.keys() | new_recipe.keys()
        if old_recipe.get(key) != new_recipe.get(key)
    }
    if differences != {"early_stopping_patience"}:
        raise ValueError(
            f"Continuation must change only patience; changed: {sorted(differences)}"
        )
    if cfg.early_stopping_patience <= parent_cfg["early_stopping_patience"]:
        raise ValueError(
            "Continuation must increase patience and retain the original epoch ceiling."
        )
    if (
        not cfg.auto_resume
        or not cfg.publication_eval_enabled
        or cfg.final_test_eval_enabled
    ):
        raise ValueError(
            "Continuation requires publication evaluation, auto-resume, and locked test evaluation."
        )
    if cfg.continuation_lr_schedule != "original" or cfg.lr_scheduler_type != "cosine":
        raise ValueError(
            "Only the explicitly retained original cosine schedule is supported."
        )
    checkpoints = list((source / "finetune").glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        raise ValueError(f"No retained fine-tuning checkpoint: {source}")
    latest = max(checkpoints, key=lambda p: int(p.parent.name.split("-")[-1]))
    state = _json(latest)
    checkpoint = Path(state["best_model_checkpoint"]).resolve()
    if checkpoint.parent != (source / "finetune").resolve():
        raise ValueError(
            "Parent checkpoint points outside its own fine-tuning directory."
        )
    state = _validate_checkpoint(checkpoint)
    if (
        state.get("best_global_step") != state["global_step"]
        or state.get("best_metric") is None
    ):
        raise ValueError(
            "Continuation must fork a retained best checkpoint with a known selection metric."
        )
    if state["num_train_epochs"] != parent_cfg["num_train_epochs"]:
        raise ValueError(
            "Parent scheduler horizon disagrees with its frozen configuration."
        )
    if state["epoch"] <= 0 or state["epoch"] != int(state["epoch"]):
        raise ValueError("Continuation requires an epoch-boundary best checkpoint.")
    return {
        "source": str(source),
        "destination": str(destination),
        "checkpoint": str(checkpoint),
        "parent_state_dir": str(state_dir),
        "parent_wandb_run": cfg.continuation_parent_wandb_run,
        "anchor_epoch": state["epoch"],
        "anchor_step": state["global_step"],
        "original_max_steps": state["max_steps"],
        "original_epochs": state["num_train_epochs"],
        "original_patience": parent_cfg["early_stopping_patience"],
        "target_epochs": cfg.num_train_epochs,
        "target_patience": cfg.early_stopping_patience,
        "lr_schedule": cfg.continuation_lr_schedule,
        "recipe_digest": content_digest(new_recipe),
        "parent_contract": contract,
    }


def prepare_continuation(cfg: Config) -> dict[str, Any]:
    """Copy a completed checkpoint atomically; never overwrite or hardlink parent data."""
    info = inspect_parent(cfg)
    destination = Path(info["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination.parent / ".continuation-prepare.lock"), timeout=0):
        if destination.exists():
            return validate_continuation(cfg)
        stage = destination.parent / f".{destination.name}.prepare-{uuid4().hex}"
        stage.mkdir(mode=0o700)
        checkpoint = Path(info["checkpoint"])
        copied_checkpoint = stage / "finetune" / checkpoint.name
        shutil.copytree(checkpoint, copied_checkpoint, copy_function=shutil.copy2)
        hashes = {p.name: _sha256(p) for p in checkpoint.iterdir()}
        if any(
            _sha256(copied_checkpoint / name) != digest
            for name, digest in hashes.items()
        ):
            raise ValueError(
                f"Copy verification failed; untouched parent retained, inspect {stage}"
            )
        trainer_state = _json(copied_checkpoint / "trainer_state.json")
        trainer_state["best_model_checkpoint"] = str(
            destination / "finetune" / checkpoint.name
        )
        trainer_state["log_history"] = []
        trainer_state["stateful_callbacks"] = {}
        write_json(copied_checkpoint / "trainer_state.json", trainer_state)
        for name in ("wandb_run_id.txt",):
            if (copied_checkpoint / name).exists():
                raise ValueError(
                    f"Unexpected tracking identity in checkpoint; inspect {stage}"
                )
        publication = stage / "publication"
        publication.mkdir(mode=0o700)
        source_publication = Path(info["source"]) / "publication"
        shutil.copytree(source_publication / "manifests", publication / "manifests")
        shutil.copy2(
            source_publication / "reference.json", publication / "reference.json"
        )
        contract = dict(
            info["parent_contract"], training_recipe_digest=info["recipe_digest"]
        )
        write_json(publication / "contract.json", contract)
        write_json(publication / "effective_config.json", configuration_payload(cfg))
        info["checkpoint_sha256"] = hashes
        info["initial_trainer_state_sha256"] = _sha256(
            copied_checkpoint / "trainer_state.json"
        )
        info["wandb_run_id"] = uuid4().hex
        info["version"] = 1
        info["fingerprint"] = content_digest(info)
        write_json(stage / "continuation.json", info)
        (stage / "wandb_run_id.txt").write_text(info["wandb_run_id"] + "\n")
        (stage / "wandb_run_id.txt").chmod(0o600)
        current_parent = inspect_parent(cfg)
        if any(current_parent[key] != info[key] for key in current_parent):
            raise ValueError(f"Parent changed during preparation; inspect {stage}")
        stage.rename(destination)
        return info


def validate_continuation(cfg: Config) -> dict[str, Any]:
    """Verify a prepared branch and reject foreign identities or unsafe output reuse."""
    info = inspect_parent(cfg)
    destination = Path(info["destination"])
    recorded = _json(destination / "continuation.json")
    if content_digest(
        {k: v for k, v in recorded.items() if k != "fingerprint"}
    ) != recorded.get("fingerprint"):
        raise ValueError("Continuation lineage manifest was modified.")
    if any(recorded.get(key) != value for key, value in info.items()):
        raise ValueError(
            "Prepared continuation no longer matches its parent or requested settings."
        )
    checkpoints = list(
        (destination / "finetune").glob("checkpoint-*/trainer_state.json")
    )
    if not checkpoints:
        raise ValueError(
            "Continuation checkpoint missing; refusing to restart from scratch."
        )
    latest = max(checkpoints, key=lambda p: int(p.parent.name.split("-")[-1]))
    state = _validate_checkpoint(latest.parent)
    best = Path(state["best_model_checkpoint"]).resolve()
    if (
        best.parent != (destination / "finetune").resolve()
        or not (best / "model.safetensors").is_file()
    ):
        raise ValueError(
            "Continuation best checkpoint is missing or points outside its child run."
        )
    if state["global_step"] < recorded["anchor_step"]:
        raise ValueError("Continuation checkpoint predates the recorded fork.")
    for path in (
        destination / "wandb_run_id.txt",
        Path(os.environ.get("CODLLM_RUN_STATE_DIR", str(destination)))
        / "wandb_run_id.txt",
    ):
        if path.exists() and path.read_text().strip() != recorded["wandb_run_id"]:
            raise ValueError(f"Foreign W&B run identity: {path}")
    if (
        os.environ.get("WANDB_RUN_ID", recorded["wandb_run_id"])
        != recorded["wandb_run_id"]
    ):
        raise ValueError(
            "WANDB_RUN_ID must not override the new continuation identity."
        )
    for name, expected in (
        ("WANDB_ENTITY", cfg.wandb.entity),
        ("WANDB_PROJECT", cfg.wandb.project),
    ):
        if os.environ.get(name, expected) != expected:
            raise ValueError(f"{name} must not override the continuation project.")
    return recorded
