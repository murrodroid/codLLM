from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from codllm.config import Config
from codllm.maintenance import (
    build_dataset_cache_report,
    build_git_hygiene_report,
    build_hpc_environment_report,
    clear_dataset_caches,
    write_git_snapshot,
)


def test_dataset_cache_report_and_clear_respect_dry_run_and_locks(
    tmp_path: Path,
) -> None:
    """Dataset maintenance should report and clear cache paths without raw data risk."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    processed_path = processed_dir / "data.parquet"
    metadata_path = processed_dir / "data.parquet.meta.json"
    processed_lock_path = processed_dir / "data.parquet.lock"
    splits_root = processed_dir / "data.splits"
    split_cache = splits_root / "abc123"
    split_cache.mkdir(parents=True)
    split_lock_path = splits_root / "abc123.lock"
    temporary_path = processed_dir / ".data.parquet.abc.tmp.parquet"

    processed_path.write_bytes(b"processed")
    metadata_path.write_text("{}\n", encoding="utf-8")
    processed_lock_path.write_text("", encoding="utf-8")
    (split_cache / "train.parquet").write_bytes(b"train")
    split_lock_path.write_text("", encoding="utf-8")
    temporary_path.write_bytes(b"tmp")

    cfg = Config(
        data_processed_dir=str(processed_dir), processed_filename="data.parquet"
    )
    report = build_dataset_cache_report(cfg)
    kinds = {entry.kind for entry in report.entries if entry.exists}

    assert "processed" in kinds
    assert "prepared_splits_cache" in kinds
    assert "prepared_splits_lock" in kinds
    assert "temporary" in kinds
    assert report.total_size_bytes > 0

    dry_run_actions = clear_dataset_caches(cfg, locks=False, execute=False)

    assert any(action.kind == "prepared_splits_cache" for action in dry_run_actions)
    assert processed_path.exists()
    assert split_cache.exists()

    executed_actions = clear_dataset_caches(cfg, locks=False, execute=True)

    assert any(action.executed for action in executed_actions)
    assert not processed_path.exists()
    assert not metadata_path.exists()
    assert not split_cache.exists()
    assert not temporary_path.exists()
    assert processed_lock_path.exists()
    assert split_lock_path.exists()


def test_dataset_cache_clear_rejects_paths_outside_processed_dir(
    tmp_path: Path,
) -> None:
    """Dataset cache cleanup should reject processed filenames that escape the cache root."""
    cfg = Config(
        data_processed_dir=str(tmp_path / "processed"),
        processed_filename="../outside.parquet",
    )

    with pytest.raises(ValueError, match="outside data_processed_dir"):
        clear_dataset_caches(cfg, execute=False)


def test_hpc_environment_report_warns_for_unset_and_outside_paths(
    tmp_path: Path,
) -> None:
    """HPC env maintenance should flag missing storage setup and leaked caches."""
    missing = build_hpc_environment_report({})

    assert missing.has_issues
    assert missing.issues[0].key == "RUN_STORAGE_DIR"

    storage_root = tmp_path / "run-storage"
    report = build_hpc_environment_report(
        {
            "RUN_STORAGE_DIR": str(storage_root),
            "UV_CACHE_DIR": str(storage_root / "cache/uv"),
            "HF_HOME": str(tmp_path / "home/.cache/huggingface"),
            "XDG_CACHE_HOME": str(storage_root / "cache/xdg"),
            "XDG_CACHE_HOME_DIR": str(storage_root / "cache/other-xdg"),
        },
        uv_cache_dir=str(tmp_path / "home/.cache/uv"),
    )
    issue_keys = {issue.key for issue in report.issues}

    assert "HF_HOME" in issue_keys
    assert "uv cache dir" in issue_keys
    assert "XDG_CACHE_HOME" in issue_keys


def test_git_hygiene_report_checks_generated_ignore_probes(tmp_path: Path) -> None:
    """Git hygiene should report generated probes that are not ignored."""
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("logs/\n", encoding="utf-8")

    report = build_git_hygiene_report(
        tmp_path,
        generated_probes=("logs/_probe", "not_ignored/_probe"),
        status_paths=(),
    )

    assert report.ignored_probes["logs/_probe"] is True
    assert report.ignored_probes["not_ignored/_probe"] is False
    assert report.has_issues
    assert report.issues[0].path == "not_ignored/_probe"


def test_write_git_snapshot_creates_local_log(tmp_path: Path) -> None:
    """Git snapshots should capture status even before the first commit."""
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("logs/\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("local\n", encoding="utf-8")

    snapshot_path = write_git_snapshot(tmp_path, max_commits=2)
    content = snapshot_path.read_text(encoding="utf-8")

    assert snapshot_path.exists()
    assert snapshot_path.parent == tmp_path / "logs/git"
    assert "## status" in content
    assert "untracked.txt" in content


def _git(repo_dir: Path, *args: str) -> None:
    """Run one git command in a temporary test repository."""
    subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True)
