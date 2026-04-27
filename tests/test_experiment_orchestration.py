from pathlib import Path

from codllm.experiments import (
    LsfProfile,
    format_env_file,
    load_experiment_spec,
    load_lsf_profiles,
    prepare_lsf_submission,
)
from tasks import _profile_for_lsf_user


def test_experiment_spec_inherits_env_and_expands_cartesian_sweep(
    tmp_path: Path,
) -> None:
    """Experiment specs should inherit base env values and expand all sweep combinations."""
    base_path = tmp_path / "base.toml"
    base_path.write_text(
        """
name = "base"

[env]
CODLLM_HF_MODEL = "google/flan-t5-small"
CODLLM_TRAINING_INPUT = ["cod", "age"]
CODLLM_VERBOSE = true
""",
        encoding="utf-8",
    )
    spec_path = tmp_path / "sweep.toml"
    spec_path.write_text(
        """
base = "base.toml"
name = "sweep"
command = "train"
force_reprocess = true

[env]
CODLLM_HF_MODEL = "google/flan-t5-large"

[sweep]
CODLLM_LR = [1e-5, 2.5e-5]
CODLLM_NUM_TRAIN_EPOCHS = [1, 2]
""",
        encoding="utf-8",
    )

    spec = load_experiment_spec(spec_path)
    runs = spec.expanded_runs()

    assert spec.env["CODLLM_HF_MODEL"] == "google/flan-t5-large"
    assert spec.env["CODLLM_TRAINING_INPUT"] == "cod,age"
    assert spec.env["CODLLM_VERBOSE"] == "1"
    assert len(runs) == 4
    assert runs[0].force_reprocess is True
    assert runs[0].env["CODLLM_LR"] == "1e-05"
    assert runs[0].env["CODLLM_NUM_TRAIN_EPOCHS"] == "1"


def test_format_env_file_includes_runtime_metadata(tmp_path: Path) -> None:
    """Generated env files should include explicit orchestration metadata."""
    spec_path = tmp_path / "run.toml"
    spec_path.write_text(
        """
name = "run"

[env]
CODLLM_HF_MODEL = "google/flan-t5-small"
""",
        encoding="utf-8",
    )
    spec = load_experiment_spec(spec_path)
    run = spec.expanded_runs()[0]

    content = format_env_file(run, spec)

    assert "export CODLLM_EXPERIMENT_NAME=run" in content
    assert "export CODLLM_JOB_COMMAND=train" in content
    assert "export CODLLM_WANDB_RUN_NAME=run" in content


def test_prepare_lsf_submission_writes_script_env_files_and_manifest(
    tmp_path: Path,
) -> None:
    """LSF generation should write a submit script, env files, and manifest."""
    spec_path = tmp_path / "sweep.toml"
    spec_path.write_text(
        """
name = "sweep"

[env]
CODLLM_HF_MODEL = "google/flan-t5-small"

[sweep]
CODLLM_NUM_TRAIN_EPOCHS = [1, 2]
""",
        encoding="utf-8",
    )
    spec = load_experiment_spec(spec_path)
    profile = LsfProfile(
        name="test",
        queue="gpu",
        wall_time="00:30",
        cores=2,
        memory="2GB",
        gpu="num=1:mode=exclusive_process",
        email=None,
        modules=("cuda/12.2",),
        sync_env=False,
    )

    submission = prepare_lsf_submission(
        spec,
        profile,
        project_dir=tmp_path,
        output_root="jobs/generated",
    )

    assert submission.script_path.exists()
    assert submission.manifest_path.exists()
    assert (submission.env_dir / "run-1.env").exists()
    assert (submission.env_dir / "run-2.env").exists()
    script = submission.script_path.read_text(encoding="utf-8")
    env_file = (submission.env_dir / "run-2.env").read_text(encoding="utf-8")
    manifest = submission.manifest_path.read_text(encoding="utf-8")

    assert '#BSUB -J "codllm-sweep[1-2]"' in script
    assert "module load cuda/12.2" in script
    assert "uv run python -m codllm.training" in script
    assert "export CODLLM_EXPERIMENT_SWEEP_INDEX=2" in env_file
    assert '"run_count": 2' in manifest


def test_training_inputs_spec_only_sets_sweep_overrides() -> None:
    """Input-feature sweep should leave config-default training choices implicit."""
    spec = load_experiment_spec("runs/sweeps/training_inputs.toml")

    assert spec.env["CODLLM_LR"] == "2.5e-05"
    assert spec.env["CODLLM_NUM_TRAIN_EPOCHS"] == "8"
    assert spec.env["CODLLM_SAVE_STRATEGY"] == "no"

    for key in {
        "CODLLM_MODEL_TASK",
        "CODLLM_DATASET_SIZE",
        "CODLLM_BALANCE_STRATEGY",
        "CODLLM_WARMUP_STEPS",
        "CODLLM_TRAIN_SIZE",
        "CODLLM_VAL_SIZE",
        "CODLLM_TEST_SIZE",
        "CODLLM_PRETRAIN_ENABLED",
    }:
        assert key not in spec.env


def test_h100_base_leaves_model_dtype_to_config_default() -> None:
    """H100 runtime bases should not force model-load precision."""
    large = load_experiment_spec("runs/base/h100-large.toml")
    small = load_experiment_spec("runs/base/h100-small.toml")

    assert "CODLLM_TORCH_DTYPE" not in large.env
    assert "CODLLM_TORCH_DTYPE" not in small.env


def test_h100_base_uses_high_throughput_dataloader_settings() -> None:
    """H100 runtime bases should request enough input pipeline capacity."""
    large = load_experiment_spec("runs/base/h100-large.toml")
    small = load_experiment_spec("runs/base/h100-small.toml")

    for spec in (large, small):
        assert spec.env["CODLLM_DATALOADER_NUM_WORKERS"] == "16"
        assert spec.env["CODLLM_DATALOADER_PERSISTENT_WORKERS"] == "1"
        assert spec.env["CODLLM_DATALOADER_PREFETCH_FACTOR"] == "2"

    assert large.env["CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE"] == "96"
    assert small.env["CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE"] == "256"


def test_h100_lsf_profiles_match_dataloader_worker_capacity() -> None:
    """H100 LSF profiles should reserve one CPU slot per worker plus the main process."""
    profiles = load_lsf_profiles("hpc/lsf_profiles.toml")

    for profile_name in ("h100-24h", "h100-10h", "h100-5h", "h100-2h"):
        assert profiles[profile_name].cores == 17


def test_profile_for_lsf_user_sets_notification_email() -> None:
    """Submit-time user aliases should override the profile email."""
    profile = LsfProfile(
        name="test",
        queue="gpu",
        wall_time="00:30",
        cores=2,
        memory="2GB",
        email="old@example.com",
    )

    updated = _profile_for_lsf_user(profile, "elias")

    assert updated.email == "s234854@dtu.dk"
    assert profile.email == "old@example.com"
