"""HPC environment maintenance and storage hygiene helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

HPC_PATH_ENV_KEYS: tuple[str, ...] = (
    "RUN_STORAGE_DIR",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON_INSTALL_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
    "TORCH_HOME",
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "XDG_CACHE_HOME",
    "XDG_CACHE_HOME_DIR",
    "VIRTUAL_ENV",
)


@dataclass(frozen=True)
class HpcEnvironmentIssue:
    """One HPC storage environment warning."""

    key: str
    value: str | None
    message: str


@dataclass(frozen=True)
class HpcEnvironmentReport:
    """Resolved HPC storage paths and any hygiene issues."""

    paths: dict[str, str | None]
    issues: tuple[HpcEnvironmentIssue, ...]

    @property
    def has_issues(self) -> bool:
        """Return True when the report contains warnings."""
        return bool(self.issues)


def build_hpc_environment_report(
    environ: Mapping[str, str],
    *,
    uv_cache_dir: str | None = None,
) -> HpcEnvironmentReport:
    """Inspect HPC storage-related environment variables."""
    paths = {key: environ.get(key) for key in HPC_PATH_ENV_KEYS}
    if uv_cache_dir is not None:
        paths["uv cache dir"] = uv_cache_dir

    issues: list[HpcEnvironmentIssue] = []
    run_storage_dir = paths.get("RUN_STORAGE_DIR")
    if run_storage_dir is None or run_storage_dir.strip() == "":
        issues.append(
            HpcEnvironmentIssue(
                key="RUN_STORAGE_DIR",
                value=run_storage_dir,
                message=(
                    "RUN_STORAGE_DIR is unset; source hpc/env.sh before uv "
                    "commands on HPC."
                ),
            )
        )
        return HpcEnvironmentReport(paths=paths, issues=tuple(issues))

    storage_root = Path(run_storage_dir).expanduser().resolve(strict=False)
    for key, value in paths.items():
        if key == "RUN_STORAGE_DIR" or value is None or value.strip() == "":
            continue
        if not _is_relative_to(Path(value).expanduser(), storage_root):
            issues.append(
                HpcEnvironmentIssue(
                    key=key,
                    value=value,
                    message=f"{key} is outside RUN_STORAGE_DIR.",
                )
            )

    xdg_cache_home = paths.get("XDG_CACHE_HOME")
    xdg_cache_home_dir = paths.get("XDG_CACHE_HOME_DIR")
    if xdg_cache_home and xdg_cache_home_dir and xdg_cache_home != xdg_cache_home_dir:
        issues.append(
            HpcEnvironmentIssue(
                key="XDG_CACHE_HOME",
                value=xdg_cache_home,
                message=(
                    "XDG_CACHE_HOME and XDG_CACHE_HOME_DIR point at different paths."
                ),
            )
        )

    return HpcEnvironmentReport(paths=paths, issues=tuple(issues))


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is below parent."""
    try:
        path.resolve(strict=False).relative_to(parent)
    except ValueError:
        return False
    return True
