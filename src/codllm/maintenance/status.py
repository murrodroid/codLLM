"""Storage status reporting for repository and HPC maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
from typing import Mapping

from codllm.config import Config
from codllm.maintenance.cache import (
    MaintenanceCacheKind,
    build_clear_cache_plan,
)
from codllm.maintenance.datasets import build_dataset_cache_report


@dataclass(frozen=True)
class StorageRootReport:
    """Disk capacity for one filesystem root relevant to codLLM."""

    name: str
    path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class StorageCategoryReport:
    """Known codLLM storage usage for one category."""

    name: str
    size_bytes: int
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class MaintenanceStatusReport:
    """Storage capacity and category usage for maintenance status output."""

    roots: tuple[StorageRootReport, ...]
    categories: tuple[StorageCategoryReport, ...]


def build_maintenance_status(
    cfg: Config,
    *,
    repo_dir: Path | str = ".",
    environ: Mapping[str, str] | None = None,
) -> MaintenanceStatusReport:
    """Build a categorized storage report for local or HPC environments."""
    environment = os.environ if environ is None else environ
    repo_path = Path(repo_dir).resolve(strict=False)
    clear_actions = build_clear_cache_plan(
        cfg,
        repo_dir=repo_path,
        environ=environment,
        mode="aggressive",
        include_env_caches=True,
    )
    cache_entries = tuple(action.entry for action in clear_actions)
    dataset_report = build_dataset_cache_report(cfg)

    categories = [
        StorageCategoryReport(
            name="code and tracked specs",
            size_bytes=_git_tracked_size(repo_path),
            paths=(repo_path,),
        ),
        StorageCategoryReport(
            name="raw datasets",
            size_bytes=_path_size(Path(cfg.data_raw_dir)),
            paths=(Path(cfg.data_raw_dir),),
        ),
        StorageCategoryReport(
            name="processed datasets and split caches",
            size_bytes=dataset_report.total_size_bytes,
            paths=(Path(cfg.data_processed_dir),),
        ),
        _cache_category("model weights and run outputs", cache_entries, "model_output"),
        _cache_category(
            "generated jobs and logs",
            cache_entries,
            "generated_job",
            "log",
        ),
        _cache_category("runtime dependency caches", cache_entries, "env_cache"),
        _cache_category(
            "Python and tool caches",
            cache_entries,
            "python_cache",
            "tool_cache",
        ),
        StorageCategoryReport(
            name="workspace total",
            size_bytes=_path_size(repo_path),
            paths=(repo_path,),
        ),
    ]

    return MaintenanceStatusReport(
        roots=_storage_roots(repo_path, environment),
        categories=tuple(categories),
    )


def _cache_category(
    name: str,
    entries: tuple,
    *kinds: MaintenanceCacheKind,
) -> StorageCategoryReport:
    """Aggregate cache entries matching one or more maintenance kinds."""
    matched_entries = tuple(entry for entry in entries if entry.kind in kinds)
    return StorageCategoryReport(
        name=name,
        size_bytes=sum(entry.size_bytes for entry in matched_entries),
        paths=tuple(entry.path for entry in matched_entries),
    )


def _storage_roots(
    repo_dir: Path,
    environ: Mapping[str, str],
) -> tuple[StorageRootReport, ...]:
    """Return capacity reports for workspace, home, and configured HPC roots."""
    candidates = [("workspace", repo_dir), ("home/profile", Path.home())]
    for key, label in (
        ("STORAGE_FOLDER", "HPC storage folder"),
        ("RUN_STORAGE_DIR", "HPC run storage"),
    ):
        value = environ.get(key)
        if value:
            candidates.append((label, Path(value).expanduser()))

    reports: list[StorageRootReport] = []
    seen_roots: set[Path] = set()
    for name, path in candidates:
        root = _existing_parent(path).resolve(strict=False)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        usage = shutil.disk_usage(root)
        reports.append(
            StorageRootReport(
                name=name,
                path=root,
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
            )
        )
    return tuple(reports)


def _existing_parent(path: Path) -> Path:
    """Return path or nearest existing parent."""
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _git_tracked_size(repo_dir: Path) -> int:
    """Return total size of git-tracked files in the repository."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_dir,
        text=False,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return 0
    total = 0
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = repo_dir / raw_path.decode("utf-8", errors="replace")
        if path.exists() and path.is_file():
            total += path.stat().st_size
    return total


def _path_size(path: Path) -> int:
    """Return file or directory size without following symlinked directories."""
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink() or child.is_file():
            total += child.lstat().st_size
    return total
