import os
import subprocess
from pathlib import Path

import pytest


def _create_fake_uv(tmp_path: Path) -> None:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'echo "FAKE_UV_ARGS:$*"\n'
        'echo "ENV_CODLLM_HF_MODEL:${CODLLM_HF_MODEL:-}"\n'
        'echo "ENV_CODLLM_LR:${CODLLM_LR:-}"\n'
        'echo "ENV_CODLLM_NUM_TRAIN_EPOCHS:${CODLLM_NUM_TRAIN_EPOCHS:-}"\n'
        'echo "ENV_CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE:${CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE:-}"\n'
        'echo "ENV_CODLLM_PER_DEVICE_EVAL_BATCH_SIZE:${CODLLM_PER_DEVICE_EVAL_BATCH_SIZE:-}"\n'
        'echo "ENV_CODLLM_WEIGHT_DECAY:${CODLLM_WEIGHT_DECAY:-}"\n'
        'echo "ENV_CODLLM_TRAINING_INPUT:${CODLLM_TRAINING_INPUT:-}"\n'
        'echo "ENV_CODLLM_DATA_PROCESSED_DIR:${CODLLM_DATA_PROCESSED_DIR:-}"\n'
        "exit 0\n"
    )
    fake_uv.chmod(0o755)


def _create_fake_bsub(tmp_path: Path) -> None:
    fake_bsub = tmp_path / "bsub"
    fake_bsub.write_text(
        '#!/usr/bin/env bash\necho "FAKE_BSUB_ARGS:$*"\ncat >/dev/null\nexit 0\n'
    )
    fake_bsub.chmod(0o755)


def test_train_script_loads_job_config_file(tmp_path: Path) -> None:
    """jobs/train.sh should load JOB_CONFIG_FILE values into the training process."""
    if os.name == "nt":
        pytest.skip("jobs/train.sh is a bash script and is not supported on Windows.")

    _create_fake_uv(tmp_path)

    job_config_file = tmp_path / "job.env"
    job_config_file.write_text(
        "CODLLM_HF_MODEL=google/flan-t5-base\n"
        "CODLLM_LR=9e-5\n"
        "CODLLM_NUM_TRAIN_EPOCHS=9\n"
        "CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE=3\n"
        "CODLLM_PER_DEVICE_EVAL_BATCH_SIZE=2\n"
        "CODLLM_WEIGHT_DECAY=0.123\n"
        "CODLLM_TRAINING_INPUT=cod,age\n"
        "CODLLM_DATA_PROCESSED_DIR=/tmp/custom-processed\n"
        "SYNC_ENV=0\n"
    )

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("REQUIRE_JOB_CONFIG_FILE", None)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["JOB_CONFIG_FILE"] = str(job_config_file)
    env["RUN_STORAGE_DIR"] = str(tmp_path / "run-storage")

    result = subprocess.run(
        ["bash", "jobs/train.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Loading job config file" in result.stdout
    assert "FAKE_UV_ARGS:run python -m codllm.train" in result.stdout
    assert "ENV_CODLLM_HF_MODEL:google/flan-t5-base" in result.stdout
    assert "ENV_CODLLM_LR:9e-5" in result.stdout
    assert "ENV_CODLLM_NUM_TRAIN_EPOCHS:9" in result.stdout
    assert "ENV_CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE:3" in result.stdout
    assert "ENV_CODLLM_PER_DEVICE_EVAL_BATCH_SIZE:2" in result.stdout
    assert "ENV_CODLLM_WEIGHT_DECAY:0.123" in result.stdout
    assert "ENV_CODLLM_TRAINING_INPUT:cod,age" in result.stdout
    assert "ENV_CODLLM_DATA_PROCESSED_DIR:/tmp/custom-processed" in result.stdout


def test_train_script_resolves_job_config_from_jobs_configs_dir(tmp_path: Path) -> None:
    """jobs/train.sh should resolve bare JOB_CONFIG_FILE names from jobs/configs."""
    if os.name == "nt":
        pytest.skip("jobs/train.sh is a bash script and is not supported on Windows.")

    _create_fake_uv(tmp_path)

    project_dir = tmp_path / "project"
    (project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    config_dir = project_dir / "jobs" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "example.env").write_text(
        "CODLLM_LR=7e-5\nCODLLM_NUM_TRAIN_EPOCHS=7\nSYNC_ENV=0\n"
    )

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("REQUIRE_JOB_CONFIG_FILE", None)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["LSB_SUBCWD"] = str(project_dir)
    env["JOB_CONFIG_FILE"] = "example.env"
    env["RUN_STORAGE_DIR"] = str(tmp_path / "run-storage")

    result = subprocess.run(
        ["bash", "jobs/train.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Loading job config file: {config_dir / 'example.env'}" in result.stdout
    assert "ENV_CODLLM_LR:7e-5" in result.stdout
    assert "ENV_CODLLM_NUM_TRAIN_EPOCHS:7" in result.stdout


def test_train_script_requires_job_config_when_flag_enabled(tmp_path: Path) -> None:
    """jobs/train.sh should fail early when REQUIRE_JOB_CONFIG_FILE=1 and no config is provided."""
    if os.name == "nt":
        pytest.skip("jobs/train.sh is a bash script and is not supported on Windows.")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["REQUIRE_JOB_CONFIG_FILE"] = "1"
    env.pop("JOB_CONFIG_FILE", None)

    result = subprocess.run(
        ["bash", "jobs/train.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "REQUIRE_JOB_CONFIG_FILE=1 but JOB_CONFIG_FILE is not set." in result.stdout


def test_train_h100_submit_mode_applies_runtime_override(tmp_path: Path) -> None:
    """jobs/train_h100.sh submit mode should map RUNTIME env config to bsub -W."""
    if os.name == "nt":
        pytest.skip(
            "jobs/train_h100.sh is a bash script and is not supported on Windows."
        )

    _create_fake_bsub(tmp_path)

    job_config_file = tmp_path / "job-h100.env"
    job_config_file.write_text("RUNTIME=02:00\n")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        ["bash", "jobs/train_h100.sh", "--submit", str(job_config_file)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Loading job config file" in result.stdout
    assert "Submitting LSF job with command:" in result.stdout
    assert "FAKE_BSUB_ARGS:" in result.stdout
    assert "-W 02:00" in result.stdout
    assert "JOB_CONFIG_FILE=" in result.stdout
