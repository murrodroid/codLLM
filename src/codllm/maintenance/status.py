"""Storage status reporting for repository and HPC maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
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
class StorageQuotaReport:
    """User quota usage for one storage area when exposed by the HPC environment."""

    source: str
    used_bytes: int | None
    limit_bytes: int | None
    free_bytes: int | None
    error: str | None = None


@dataclass(frozen=True)
class StorageRootReport:
    """Filesystem capacity and optional user quota for one root."""

    name: str
    path: Path
    capacity_path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    quota: StorageQuotaReport | None


@dataclass(frozen=True)
class StorageCategoryReport:
    """Storage usage for one managed or diagnostic category."""

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
    categories.extend(_hpc_storage_categories(environment, tuple(categories)))

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
    """Return filesystem and quota reports for workspace, home, and HPC roots."""
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
    quota_cache: dict[str, StorageQuotaReport | None] = {}
    for name, path in candidates:
        requested_path = path.expanduser().resolve(strict=False)
        capacity_path = _existing_parent(path).resolve(strict=False)
        if requested_path in seen_roots:
            continue
        seen_roots.add(requested_path)
        usage = shutil.disk_usage(capacity_path)
        quota = _quota_for_path(requested_path, quota_cache)
        reports.append(
            StorageRootReport(
                name=name,
                path=requested_path,
                capacity_path=capacity_path,
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                quota=quota,
            )
        )
    return tuple(reports)


def _hpc_storage_categories(
    environ: Mapping[str, str],
    known_categories: tuple[StorageCategoryReport, ...],
) -> list[StorageCategoryReport]:
    """Return HPC storage totals and uncategorized storage usage."""
    categories: list[StorageCategoryReport] = []
    run_storage = _optional_path(environ.get("RUN_STORAGE_DIR"))
    storage_folder = _optional_path(environ.get("STORAGE_FOLDER"))

    if run_storage is not None and run_storage.exists():
        run_storage_size = _path_size(run_storage)
        known_run_storage_size = _covered_category_size(run_storage, known_categories)
        categories.extend(
            [
                StorageCategoryReport(
                    name="HPC run storage total",
                    size_bytes=run_storage_size,
                    paths=(run_storage,),
                ),
                StorageCategoryReport(
                    name="uncategorized HPC run storage",
                    size_bytes=max(run_storage_size - known_run_storage_size, 0),
                    paths=(run_storage,),
                ),
            ]
        )

    if (
        storage_folder is None
        or not storage_folder.exists()
        or not storage_folder.is_dir()
    ):
        return categories

    sibling_reports: list[StorageCategoryReport] = []
    for child in sorted(storage_folder.iterdir()):
        child_path = child.resolve(strict=False)
        if run_storage is not None and child_path == run_storage.resolve(strict=False):
            continue
        size_bytes = _path_size(child)
        if size_bytes <= 0:
            continue
        sibling_reports.append(
            StorageCategoryReport(
                name=f"HPC storage sibling: {child.name}",
                size_bytes=size_bytes,
                paths=(child,),
            )
        )
    sibling_reports.sort(key=lambda report: report.size_bytes, reverse=True)
    return categories + sibling_reports[:10]


def _optional_path(value: str | None) -> Path | None:
    """Return a resolved path for a non-empty environment value."""
    if value is None or value.strip() == "":
        return None
    return Path(value).expanduser().resolve(strict=False)


def _covered_category_size(
    root: Path,
    categories: tuple[StorageCategoryReport, ...],
) -> int:
    """Return known category bytes covered by paths below one root."""
    covered_size = 0
    covered_dirs: list[Path] = []
    root_path = root.resolve(strict=False)
    category_paths = sorted(
        (
            path
            for category in categories
            for path in category.paths
            if _is_relative_to(path.resolve(strict=False), root_path)
        ),
        key=lambda path: len(path.resolve(strict=False).parts),
    )
    for path in category_paths:
        resolved_path = path.resolve(strict=False)
        if any(
            _is_relative_to(resolved_path, covered_dir) for covered_dir in covered_dirs
        ):
            continue
        covered_size += _path_size(path)
        if path.is_dir() and not path.is_symlink():
            covered_dirs.append(resolved_path)
    return covered_size


def _existing_parent(path: Path) -> Path:
    """Return path or nearest existing parent."""
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _quota_for_path(
    path: Path,
    quota_cache: dict[str, StorageQuotaReport | None],
) -> StorageQuotaReport | None:
    """Return DTU HPC quota information for a path when the relevant command exists."""
    command = _quota_command_for_path(path)
    if command is None:
        return None
    source = " ".join(command)
    if source not in quota_cache:
        quota_cache[source] = _quota_from_command(command)
    return quota_cache[source]


def _quota_command_for_path(path: Path) -> tuple[str, ...] | None:
    """Return the DTU quota command for known storage roots."""
    parts = path.expanduser().resolve(strict=False).parts
    if len(parts) < 2:
        return None
    root_name = parts[1]
    if root_name == "zhome":
        return ("getquota_zhome.sh",)
    if root_name in {"work1", "work3"}:
        return (f"getquota_{root_name}.sh",)
    return None


def _quota_from_command(command: tuple[str, ...]) -> StorageQuotaReport | None:
    """Run and parse one quota command if it is installed."""
    if shutil.which(command[0]) is None:
        return None
    source = " ".join(command)
    try:
        result = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return StorageQuotaReport(
            source=source,
            used_bytes=None,
            limit_bytes=None,
            free_bytes=None,
            error=str(exc),
        )

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0:
        return StorageQuotaReport(
            source=source,
            used_bytes=None,
            limit_bytes=None,
            free_bytes=None,
            error=output or f"command exited with code {result.returncode}",
        )

    parsed = _parse_quota_output(output)
    if parsed is None:
        return StorageQuotaReport(
            source=source,
            used_bytes=None,
            limit_bytes=None,
            free_bytes=None,
            error="could not parse command output",
        )
    used_bytes, limit_bytes = parsed
    return StorageQuotaReport(
        source=source,
        used_bytes=used_bytes,
        limit_bytes=limit_bytes,
        free_bytes=max(limit_bytes - used_bytes, 0),
    )


def _parse_quota_output(output: str) -> tuple[int, int] | None:
    """Parse supported DTU quota command output."""
    simple_match = re.search(
        r"using\s+"
        r"(?P<used>[0-9]+(?:\.[0-9]+)?)\s*(?P<used_unit>[kmgtpe]?i?b?)"
        r"\s+of\s+"
        r"(?P<limit>[0-9]+(?:\.[0-9]+)?)\s*(?P<limit_unit>[kmgtpe]?i?b?)",
        output,
        flags=re.IGNORECASE,
    )
    if simple_match:
        used_bytes = _parse_byte_value(
            simple_match.group("used"),
            simple_match.group("used_unit"),
        )
        limit_bytes = _parse_byte_value(
            simple_match.group("limit"),
            simple_match.group("limit_unit"),
        )
        return (used_bytes, limit_bytes)

    table_match = re.search(
        r"\|\|\s*"
        r"(?P<used>[0-9]+(?:\.[0-9]+)?)\s*(?P<used_unit>[kmgtpe]i?b|[kmgtpe]b?)"
        r"\s*\|\s*"
        r"(?P<limit>[0-9]+(?:\.[0-9]+)?)\s*(?P<limit_unit>[kmgtpe]i?b|[kmgtpe]b?)"
        r"\s*\|\|",
        output,
        flags=re.IGNORECASE,
    )
    if table_match:
        used_bytes = _parse_byte_value(
            table_match.group("used"),
            table_match.group("used_unit"),
        )
        limit_bytes = _parse_byte_value(
            table_match.group("limit"),
            table_match.group("limit_unit"),
        )
        return (used_bytes, limit_bytes)
    return None


def _parse_byte_value(value: str, unit: str) -> int:
    """Parse a quota byte value with decimal or binary units."""
    normalized_unit = unit.strip().lower()
    if normalized_unit in {"", "b"}:
        multiplier = 1
    else:
        prefix = normalized_unit[0]
        if prefix not in "kmgtpe":
            return int(float(value))
        base = 1024 if "i" in normalized_unit else 1000
        multiplier = base ** ("kmgtpe".index(prefix) + 1)
    return int(float(value) * multiplier)


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
        try:
            if child.is_symlink() or child.is_file():
                total += child.lstat().st_size
        except OSError:
            continue
    return total


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or below parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
