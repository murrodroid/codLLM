"""Broad generated-cache maintenance across datasets, runs, logs, and env caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import time
from typing import Literal, Mapping

from codllm.config import Config
from codllm.maintenance.datasets import clear_dataset_caches
from codllm.run_directory import LOCAL_RUN_DIR_PATTERN

CacheClearMode = Literal["standard", "aggressive"]
MaintenanceCacheKind = Literal[
    "dataset_cache",
    "generated_job",
    "log",
    "model_output",
    "python_cache",
    "tool_cache",
    "env_cache",
]

ENV_CACHE_KEYS: tuple[str, ...] = (
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
    "PIP_CACHE_DIR",
)


@dataclass(frozen=True)
class MaintenanceCacheEntry:
    """One generated cache or output path eligible for maintenance cleanup."""

    path: Path
    kind: MaintenanceCacheKind
    size_bytes: int
    last_activity_ns: int
    reason: str


@dataclass(frozen=True)
class MaintenanceCacheAction:
    """One planned or executed broad cache cleanup action."""

    entry: MaintenanceCacheEntry
    executed: bool


def build_clear_cache_plan(
    cfg: Config,
    *,
    repo_dir: Path | str = ".",
    environ: Mapping[str, str] | None = None,
    mode: CacheClearMode = "standard",
    retention_days: int = 14,
    locks: bool = False,
    include_env_caches: bool = True,
) -> tuple[MaintenanceCacheAction, ...]:
    """Return broad cache cleanup actions without deleting anything."""
    if retention_days < 0:
        raise ValueError("retention_days must be non-negative.")
    environment = os.environ if environ is None else environ
    repo_path = Path(repo_dir).resolve(strict=False)
    cutoff_ns = time.time_ns() - retention_days * 24 * 60 * 60 * 1_000_000_000
    entries = _discover_generated_cache_entries(
        cfg,
        repo_dir=repo_path,
        environ=environment,
        locks=locks,
        include_env_caches=include_env_caches,
    )
    if mode == "standard":
        entries = tuple(
            entry for entry in entries if entry.last_activity_ns <= cutoff_ns
        )
    elif mode != "aggressive":
        raise ValueError("mode must be either 'standard' or 'aggressive'.")

    return tuple(
        MaintenanceCacheAction(entry=entry, executed=False) for entry in entries
    )


def clear_generated_caches(
    cfg: Config,
    *,
    repo_dir: Path | str = ".",
    environ: Mapping[str, str] | None = None,
    mode: CacheClearMode = "standard",
    retention_days: int = 14,
    locks: bool = False,
    include_env_caches: bool = True,
    execute: bool = False,
) -> tuple[MaintenanceCacheAction, ...]:
    """Plan or execute broad generated-cache cleanup."""
    actions = build_clear_cache_plan(
        cfg,
        repo_dir=repo_dir,
        environ=environ,
        mode=mode,
        retention_days=retention_days,
        locks=locks,
        include_env_caches=include_env_caches,
    )
    if not execute:
        return actions

    executed_actions: list[MaintenanceCacheAction] = []
    for action in actions:
        _remove_path(action.entry.path)
        executed_actions.append(
            MaintenanceCacheAction(entry=action.entry, executed=True)
        )
    return tuple(executed_actions)


def _discover_generated_cache_entries(
    cfg: Config,
    *,
    repo_dir: Path,
    environ: Mapping[str, str],
    locks: bool,
    include_env_caches: bool,
) -> tuple[MaintenanceCacheEntry, ...]:
    """Return generated paths that are safe for maintenance cleanup."""
    entries: list[MaintenanceCacheEntry] = []
    safe_roots = _safe_roots(repo_dir, environ)
    for action in clear_dataset_caches(
        cfg,
        processed=True,
        splits=True,
        locks=locks,
        temporary=True,
        execute=False,
    ):
        if not action.exists:
            continue
        entries.append(
            _entry(
                action.path,
                "dataset_cache",
                safe_roots,
                "processed data, metadata, prepared split cache, or temp file",
            )
        )

    for path, kind, reason in _repo_generated_paths(repo_dir, cfg, environ):
        if path.exists() or path.is_symlink():
            entries.append(_entry(path, kind, safe_roots, reason))

    if include_env_caches:
        for path in _env_cache_paths(environ, locks=locks):
            if path.exists() or path.is_symlink():
                entries.append(
                    _entry(path, "env_cache", safe_roots, "runtime dependency cache")
                )

    return _dedupe_entries(entries)


def _repo_generated_paths(
    repo_dir: Path,
    cfg: Config,
    environ: Mapping[str, str],
) -> tuple[tuple[Path, MaintenanceCacheKind, str], ...]:
    """Return generated repo-local paths that can be removed safely."""
    paths: list[tuple[Path, MaintenanceCacheKind, str]] = []
    for root, kind, reason in (
        (repo_dir / "jobs/generated", "generated_job", "generated LSF submission"),
        (repo_dir / "logs", "log", "local or HPC log output"),
        (repo_dir / "runs_smoke", "model_output", "local smoke-test model output"),
    ):
        paths.extend(
            (child, kind, reason)
            for child in _children(root)
            if not _is_git_keep(child)
        )

    output_roots = _model_output_roots(repo_dir, cfg)
    for output_root in output_roots:
        paths.extend(
            (run_dir, "model_output", "training run output with checkpoints/weights")
            for run_dir in _run_dirs(output_root)
        )

    excluded_python_cache_roots = _excluded_python_cache_roots(repo_dir, environ)
    paths.extend(
        (path, "python_cache", "Python bytecode cache")
        for path in repo_dir.rglob("__pycache__")
        if path.is_dir()
        and not any(_is_relative_to(path, root) for root in excluded_python_cache_roots)
    )
    for tool_cache in (repo_dir / ".pytest_cache", repo_dir / ".ruff_cache"):
        if tool_cache.exists():
            paths.append((tool_cache, "tool_cache", "local development tool cache"))
    return tuple(paths)


def _excluded_python_cache_roots(
    repo_dir: Path,
    environ: Mapping[str, str],
) -> tuple[Path, ...]:
    """Return virtualenv roots excluded from Python bytecode cleanup."""
    roots = [repo_dir / ".venv"]
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        value = environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    return tuple(root.resolve(strict=False) for root in roots)


def _model_output_roots(repo_dir: Path, cfg: Config) -> tuple[Path, ...]:
    """Return likely roots containing run-* training output directories."""
    roots = [Path(cfg.output_dir)]
    if not roots[0].is_absolute():
        roots[0] = repo_dir / roots[0]
    roots.extend([repo_dir / "runs", repo_dir / "runs_smoke"])
    return tuple(_dedupe_paths(root.resolve(strict=False) for root in roots))


def _run_dirs(output_root: Path) -> tuple[Path, ...]:
    """Return run-* directories under an output root and one model-name level below."""
    if not output_root.exists():
        return ()
    run_dirs = [
        child
        for child in output_root.iterdir()
        if child.is_dir() and LOCAL_RUN_DIR_PATTERN.match(child.name)
    ]
    for child in output_root.iterdir():
        if not child.is_dir() or LOCAL_RUN_DIR_PATTERN.match(child.name):
            continue
        run_dirs.extend(
            grandchild
            for grandchild in child.iterdir()
            if grandchild.is_dir() and LOCAL_RUN_DIR_PATTERN.match(grandchild.name)
        )
    return tuple(sorted(run_dirs))


def _env_cache_paths(environ: Mapping[str, str], *, locks: bool) -> tuple[Path, ...]:
    """Return explicit runtime cache roots from the current environment."""
    paths: list[Path] = []
    for key in ENV_CACHE_KEYS:
        value = environ.get(key)
        if value is None or value.strip() == "":
            continue
        path = Path(value).expanduser().resolve(strict=False)
        if key in {"UV_PROJECT_ENVIRONMENT", "UV_PYTHON_INSTALL_DIR"} and not (
            _looks_codllm_owned_cache(path, environ)
        ):
            continue
        if key == "XDG_CACHE_HOME" and not _looks_codllm_owned_cache(path, environ):
            continue
        paths.append(path)
    paths.extend(_default_runtime_cache_paths(environ, locks=locks))
    return tuple(_dedupe_paths(paths))


def _default_runtime_cache_paths(
    environ: Mapping[str, str],
    *,
    locks: bool,
) -> tuple[Path, ...]:
    """Return managed runtime paths implied by RUN_STORAGE_DIR."""
    run_storage_dir = environ.get("RUN_STORAGE_DIR")
    if run_storage_dir is None or run_storage_dir.strip() == "":
        return ()

    root = Path(run_storage_dir).expanduser().resolve(strict=False)
    paths = [
        root / "cache",
        root / ".venv",
        root / "python",
    ]
    if locks:
        lock_file = environ.get("UV_SYNC_LOCK_FILE")
        paths.append(
            Path(lock_file).expanduser() if lock_file else root / ".uv-sync.lock"
        )
    return tuple(paths)


def _looks_codllm_owned_cache(path: Path, environ: Mapping[str, str]) -> bool:
    """Return True when a cache root appears to belong to this project runtime."""
    run_storage_dir = environ.get("RUN_STORAGE_DIR")
    if run_storage_dir and _is_relative_to(
        path,
        Path(run_storage_dir).expanduser().resolve(strict=False),
    ):
        return True
    return "codllm" in {part.lower() for part in path.parts}


def _safe_roots(repo_dir: Path, environ: Mapping[str, str]) -> tuple[Path, ...]:
    """Return roots under which maintenance is allowed to remove generated paths."""
    roots = [repo_dir, Path.home().resolve(strict=False)]
    for key in ("RUN_STORAGE_DIR", "STORAGE_FOLDER"):
        value = environ.get(key)
        if value:
            roots.append(Path(value).expanduser().resolve(strict=False))
    return tuple(_dedupe_paths(roots))


def _entry(
    path: Path,
    kind: MaintenanceCacheKind,
    safe_roots: tuple[Path, ...],
    reason: str,
) -> MaintenanceCacheEntry:
    """Build one cache entry after path-safety checks."""
    resolved_path = path.expanduser().resolve(strict=False)
    _ensure_safe_path(resolved_path, safe_roots)
    return MaintenanceCacheEntry(
        path=path,
        kind=kind,
        size_bytes=_path_size(path),
        last_activity_ns=_last_activity_ns(path),
        reason=reason,
    )


def _ensure_safe_path(path: Path, safe_roots: tuple[Path, ...]) -> None:
    """Reject broad cleanup targets outside known project/user storage roots."""
    if path == path.parent:
        raise ValueError(f"Refusing to remove filesystem root: {path}")
    for root in safe_roots:
        if path == root:
            raise ValueError(f"Refusing to remove storage root: {path}")
        if _is_relative_to(path, root):
            return
    raise ValueError(f"Refusing to remove path outside known storage roots: {path}")


def _dedupe_entries(
    entries: list[MaintenanceCacheEntry],
) -> tuple[MaintenanceCacheEntry, ...]:
    """Deduplicate nested cleanup entries, preferring parent directories."""
    sorted_entries = sorted(entries, key=lambda entry: len(entry.path.parts))
    kept: list[MaintenanceCacheEntry] = []
    kept_paths: list[Path] = []
    for entry in sorted_entries:
        resolved_path = entry.path.resolve(strict=False)
        if any(_is_relative_to(resolved_path, kept_path) for kept_path in kept_paths):
            continue
        kept.append(entry)
        if entry.path.is_dir() and not entry.path.is_symlink():
            kept_paths.append(resolved_path)
    return tuple(kept)


def _dedupe_paths(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]:
    """Return unique paths without preserving nested duplicates."""
    deduped: list[Path] = []
    for path in paths:
        resolved_path = path.resolve(strict=False)
        if resolved_path in deduped:
            continue
        if any(_is_relative_to(resolved_path, parent) for parent in deduped):
            continue
        deduped.append(resolved_path)
    return tuple(deduped)


def _children(path: Path) -> tuple[Path, ...]:
    """Return immediate children for an existing directory."""
    if not path.exists() or not path.is_dir():
        return ()
    return tuple(sorted(path.iterdir()))


def _is_git_keep(path: Path) -> bool:
    """Return True when a path is a git placeholder."""
    return path.name in {".gitkeep", ".gitignore"}


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


def _last_activity_ns(path: Path) -> int:
    """Return the newest modification timestamp below a path.

    Modification time is the reliable staleness signal: access time is not
    updated consistently across filesystems (``noatime``/``relatime``) and is
    bumped by the cleanup scan's own directory traversal, which would make
    genuinely old paths look freshly used on some CI runners.
    """
    if not path.exists() and not path.is_symlink():
        return 0
    stats = path.lstat()
    newest = stats.st_mtime_ns
    if path.is_symlink() or path.is_file():
        return newest
    for child in path.rglob("*"):
        try:
            child_stats = child.lstat()
            if child.is_symlink() or child.is_file() or child.is_dir():
                newest = max(newest, child_stats.st_mtime_ns)
        except OSError:
            continue
    return newest


def _remove_path(path: Path) -> None:
    """Remove a generated file, symlink, or directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or below parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
