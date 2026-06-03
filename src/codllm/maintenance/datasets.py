"""Maintenance helpers for processed-data and prepared-split caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Literal

from codllm.config import Config
from codllm.data import DataHandler

DatasetCacheKind = Literal[
    "processed",
    "processed_metadata",
    "processed_lock",
    "prepared_splits_root",
    "prepared_splits_cache",
    "prepared_splits_lock",
    "temporary",
]


@dataclass(frozen=True)
class DatasetCacheEntry:
    """One known dataset cache path and its current size."""

    path: Path
    kind: DatasetCacheKind
    exists: bool
    size_bytes: int


@dataclass(frozen=True)
class DatasetCacheReport:
    """Resolved dataset cache paths for the current config."""

    processed_dir: Path
    entries: tuple[DatasetCacheEntry, ...]

    @property
    def total_size_bytes(self) -> int:
        """Return total cache size while counting each filesystem subtree once."""
        total = 0
        counted_paths: list[Path] = []
        for entry in sorted(
            (entry for entry in self.entries if entry.exists),
            key=lambda item: len(item.path.parts),
        ):
            resolved_path = entry.path.resolve(strict=False)
            if any(
                _is_relative_to(resolved_path, counted) for counted in counted_paths
            ):
                continue
            total += entry.size_bytes
            if entry.path.is_dir() and not entry.path.is_symlink():
                counted_paths.append(resolved_path)
        return total


@dataclass(frozen=True)
class DatasetCacheAction:
    """One planned or executed dataset cache cleanup action."""

    path: Path
    kind: DatasetCacheKind
    exists: bool
    size_bytes: int
    executed: bool


def build_dataset_cache_report(cfg: Config) -> DatasetCacheReport:
    """Return a report of processed-data cache paths for a config."""
    handler = DataHandler(cfg)
    processed_dir = _safe_processed_dir(cfg)
    entries = [
        _cache_entry(handler.processed_path, "processed"),
        _cache_entry(handler.processed_metadata_path, "processed_metadata"),
        _cache_entry(handler.processed_lock_path, "processed_lock"),
        _cache_entry(handler.prepared_splits_root, "prepared_splits_root"),
    ]
    if handler.prepared_splits_root.exists():
        entries.extend(
            _cache_entry(path, "prepared_splits_cache")
            for path in sorted(handler.prepared_splits_root.iterdir())
            if path.is_dir()
        )
        entries.extend(
            _cache_entry(path, "prepared_splits_lock")
            for path in sorted(handler.prepared_splits_root.glob("*.lock"))
        )
    entries.extend(
        _cache_entry(path, "temporary")
        for path in _temporary_cache_paths(processed_dir, handler.prepared_splits_root)
    )
    return DatasetCacheReport(processed_dir=processed_dir, entries=tuple(entries))


def clear_dataset_caches(
    cfg: Config,
    *,
    processed: bool = True,
    splits: bool = True,
    locks: bool = False,
    temporary: bool = True,
    execute: bool = False,
) -> tuple[DatasetCacheAction, ...]:
    """Plan or execute dataset cache cleanup for the current config.

    Args:
        cfg: Runtime configuration used to resolve cache paths.
        processed: Whether to remove the processed data file and metadata sidecar.
        splits: Whether to remove prepared-split cache directories.
        locks: Whether to remove lock files. Keep this false while jobs may be active.
        temporary: Whether to remove incomplete atomic-write temporary files.
        execute: When false, only report what would be removed.

    Returns:
        Planned or executed cleanup actions.
    """
    handler = DataHandler(cfg)
    processed_dir = _safe_processed_dir(cfg)
    targets: list[tuple[Path, DatasetCacheKind]] = []

    if processed:
        targets.extend(
            [
                (handler.processed_path, "processed"),
                (handler.processed_metadata_path, "processed_metadata"),
            ]
        )
    if locks:
        targets.append((handler.processed_lock_path, "processed_lock"))
    if splits and handler.prepared_splits_root.exists():
        targets.extend(
            (path, "prepared_splits_cache")
            for path in sorted(handler.prepared_splits_root.iterdir())
            if path.is_dir()
        )
        if locks:
            targets.extend(
                (path, "prepared_splits_lock")
                for path in sorted(handler.prepared_splits_root.glob("*.lock"))
            )
    if temporary:
        targets.extend(
            (path, "temporary")
            for path in _temporary_cache_paths(
                processed_dir,
                handler.prepared_splits_root,
            )
        )

    actions: list[DatasetCacheAction] = []
    seen_paths: set[Path] = set()
    for path, kind in targets:
        _ensure_safe_cache_path(path, processed_dir)
        resolved_path = path.resolve(strict=False)
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        exists = path.exists() or path.is_symlink()
        action = DatasetCacheAction(
            path=path,
            kind=kind,
            exists=exists,
            size_bytes=_path_size(path) if exists else 0,
            executed=execute and exists,
        )
        actions.append(action)
        if execute and exists:
            _remove_path(path)
    return tuple(actions)


def format_bytes(size_bytes: int) -> str:
    """Return a compact human-readable byte count."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _cache_entry(path: Path, kind: DatasetCacheKind) -> DatasetCacheEntry:
    """Return one cache entry with existence and size metadata."""
    exists = path.exists() or path.is_symlink()
    return DatasetCacheEntry(
        path=path,
        kind=kind,
        exists=exists,
        size_bytes=_path_size(path) if exists else 0,
    )


def _safe_processed_dir(cfg: Config) -> Path:
    """Return the processed-data directory after rejecting unsafe cache paths."""
    processed_dir = Path(cfg.data_processed_dir).expanduser().resolve(strict=False)
    if processed_dir == processed_dir.parent:
        raise ValueError(
            "Config.data_processed_dir must not resolve to the filesystem root."
        )

    handler = DataHandler(cfg)
    for path in (
        handler.processed_path,
        handler.processed_metadata_path,
        handler.processed_lock_path,
        handler.prepared_splits_root,
    ):
        _ensure_safe_cache_path(path, processed_dir)
    return processed_dir


def _ensure_safe_cache_path(path: Path, processed_dir: Path) -> None:
    """Reject cache paths that escape Config.data_processed_dir."""
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(processed_dir)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to manage dataset cache path outside data_processed_dir: {path}"
        ) from exc


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or below parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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


def _remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory cache path."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _temporary_cache_paths(processed_dir: Path, splits_root: Path) -> tuple[Path, ...]:
    """Return atomic-write temporary files left behind by interrupted cache writes."""
    candidates = [
        path
        for path in processed_dir.glob(".*.tmp*")
        if path.is_file() or path.is_symlink()
    ]
    if splits_root.exists():
        candidates.extend(
            path
            for path in splits_root.rglob(".*.tmp*")
            if path.is_file() or path.is_symlink()
        )
    return tuple(sorted(candidates))
