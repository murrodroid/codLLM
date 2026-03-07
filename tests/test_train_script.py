import os
import subprocess
from pathlib import Path

import pytest


def test_train_script_loads_job_config_file(tmp_path: Path) -> None:
    """jobs/train.sh should load JOB_CONFIG_FILE values into the training process."""
    if os.name == "nt":
        pytest.skip("jobs/train.sh is a bash script and is not supported on Windows.")

    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'echo "FAKE_UV_ARGS:$*"\n'
        'echo "ENV_CODLLM_LR:${CODLLM_LR:-}"\n'
        'echo "ENV_CODLLM_NUM_TRAIN_EPOCHS:${CODLLM_NUM_TRAIN_EPOCHS:-}"\n'
        'echo "ENV_CODLLM_WEIGHT_DECAY:${CODLLM_WEIGHT_DECAY:-}"\n'
        'echo "ENV_CODLLM_TRAINING_INPUT:${CODLLM_TRAINING_INPUT:-}"\n'
        'echo "ENV_CODLLM_DATA_PROCESSED_DIR:${CODLLM_DATA_PROCESSED_DIR:-}"\n'
        "exit 0\n"
    )
    fake_uv.chmod(0o755)

    job_config_file = tmp_path / "job.env"
    job_config_file.write_text(
        "CODLLM_LR=9e-5\n"
        "CODLLM_NUM_TRAIN_EPOCHS=9\n"
        "CODLLM_WEIGHT_DECAY=0.123\n"
        "CODLLM_TRAINING_INPUT=cod,age\n"
        "CODLLM_DATA_PROCESSED_DIR=/tmp/custom-processed\n"
        "SYNC_ENV=0\n"
    )

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
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
    assert "ENV_CODLLM_LR:9e-5" in result.stdout
    assert "ENV_CODLLM_NUM_TRAIN_EPOCHS:9" in result.stdout
    assert "ENV_CODLLM_WEIGHT_DECAY:0.123" in result.stdout
    assert "ENV_CODLLM_TRAINING_INPUT:cod,age" in result.stdout
    assert "ENV_CODLLM_DATA_PROCESSED_DIR:/tmp/custom-processed" in result.stdout
