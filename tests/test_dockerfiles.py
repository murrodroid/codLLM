import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Tuple

import pytest


def _docker_is_ready() -> Tuple[bool, str]:
    """Check whether Docker CLI and daemon are available."""
    docker_path = shutil.which("docker")
    if docker_path is None:
        return False, "Docker CLI is not installed."

    probe = subprocess.run(
        [docker_path, "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        stderr = probe.stderr.strip()
        reason = stderr if stderr else "Docker daemon is unavailable."
        return False, reason

    return True, ""


def _build_dockerfile(dockerfile_path: Path) -> subprocess.CompletedProcess[str]:
    """Build one Dockerfile using the repository root as context."""
    image_tag = f"codllm-test-{dockerfile_path.stem}-{uuid.uuid4().hex[:8]}"
    try:
        return subprocess.run(
            [
                "docker",
                "build",
                "--file",
                str(dockerfile_path),
                "--tag",
                image_tag,
                ".",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        subprocess.run(
            ["docker", "image", "rm", image_tag],
            capture_output=True,
            text=True,
            check=False,
        )


DOCKER_READY, DOCKER_REASON = _docker_is_ready()


@pytest.mark.integration
@pytest.mark.skipif(not DOCKER_READY, reason=DOCKER_REASON)
def test_dockerfiles_build_successfully() -> None:
    """All Dockerfiles in dockerfiles/ should build without errors."""
    dockerfiles = sorted(Path("dockerfiles").glob("*.dockerfile"))
    assert dockerfiles, "No Dockerfiles found in dockerfiles/."

    for dockerfile_path in dockerfiles:
        result = _build_dockerfile(dockerfile_path)
        assert result.returncode == 0, (
            f"Failed to build {dockerfile_path}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
