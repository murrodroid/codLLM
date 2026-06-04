from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

import codllm.maintenance.status as status_module
from codllm.config import Config
from codllm.maintenance import (
    build_clear_cache_plan,
    build_dataset_cache_report,
    build_git_hygiene_report,
    build_hpc_environment_report,
    build_maintenance_status,
    clear_dataset_caches,
    clear_generated_caches,
    write_git_snapshot,
)
from codllm.maintenance.status import _parse_quota_output


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

    lock_actions = clear_dataset_caches(
        cfg,
        processed=False,
        splits=True,
        locks=True,
        execute=True,
    )

    assert any(action.kind == "prepared_splits_root" for action in lock_actions)
    assert not splits_root.exists()


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


def test_clear_cache_standard_filters_old_generated_paths_and_protects_raw_data(
    tmp_path: Path,
) -> None:
    """Standard broad cleanup should only target old generated maintenance paths."""
    raw_dir = tmp_path / "data/raw"
    processed_dir = tmp_path / "data/processed"
    old_run = tmp_path / "runs/run-0001"
    fresh_run = tmp_path / "runs/run-0002"
    old_job = tmp_path / "jobs/generated/old-job"
    generic_xdg_child = tmp_path / ".cache/unrelated"
    venv_pycache = tmp_path / ".venv/lib/python3.13/site-packages/pkg/__pycache__"
    for path in (
        raw_dir,
        processed_dir,
        old_run,
        fresh_run,
        old_job,
        generic_xdg_child,
        venv_pycache,
    ):
        path.mkdir(parents=True)
    (raw_dir / "source.csv").write_text("raw\n", encoding="utf-8")
    (processed_dir / "data.parquet").write_bytes(b"processed")
    (processed_dir / "data.parquet.meta.json").write_text("{}\n", encoding="utf-8")
    (old_run / "checkpoint-1").mkdir()
    (old_run / "checkpoint-1/model.safetensors").write_bytes(b"weights")
    (fresh_run / "checkpoint-1").mkdir()
    (fresh_run / "checkpoint-1/model.safetensors").write_bytes(b"fresh")
    (old_job / "submit.lsf").write_text("# job\n", encoding="utf-8")
    (generic_xdg_child / "cache.bin").write_bytes(b"cache")
    (venv_pycache / "module.pyc").write_bytes(b"pyc")

    old_timestamp = time.time() - 21 * 24 * 60 * 60
    _touch_tree(processed_dir, old_timestamp)
    _touch_tree(old_run, old_timestamp)
    _touch_tree(old_job, old_timestamp)
    _touch_tree(generic_xdg_child, old_timestamp)
    _touch_tree(tmp_path / ".venv", old_timestamp)

    cfg = Config(
        data_raw_dir=str(raw_dir),
        data_processed_dir=str(processed_dir),
        output_dir=str(tmp_path / "runs"),
        processed_filename="data.parquet",
    )
    plan = build_clear_cache_plan(
        cfg,
        repo_dir=tmp_path,
        environ={
            "XDG_CACHE_HOME": str(tmp_path / ".cache"),
            "VIRTUAL_ENV": str(tmp_path / ".venv"),
        },
        mode="standard",
        retention_days=14,
    )
    planned_paths = {action.entry.path for action in plan}

    assert processed_dir / "data.parquet" in planned_paths
    assert old_run in planned_paths
    assert old_job in planned_paths
    assert fresh_run not in planned_paths
    assert raw_dir / "source.csv" not in planned_paths
    assert generic_xdg_child not in planned_paths
    assert venv_pycache not in planned_paths


def test_clear_cache_aggressive_deletes_generated_paths(tmp_path: Path) -> None:
    """Aggressive broad cleanup should delete maintenance-managed generated paths."""
    processed_dir = tmp_path / "data/processed"
    run_dir = tmp_path / "runs/run-0001"
    processed_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (processed_dir / "data.parquet").write_bytes(b"processed")
    (run_dir / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    cfg = Config(
        data_processed_dir=str(processed_dir),
        output_dir=str(tmp_path / "runs"),
        processed_filename="data.parquet",
    )

    actions = clear_generated_caches(
        cfg,
        repo_dir=tmp_path,
        environ={},
        mode="aggressive",
        execute=True,
    )

    assert any(action.executed for action in actions)
    assert not (processed_dir / "data.parquet").exists()
    assert not run_dir.exists()


def test_clear_cache_aggressive_deletes_hpc_runtime_roots(tmp_path: Path) -> None:
    """Aggressive broad cleanup should delete regenerated HPC runtime roots."""
    storage_dir = tmp_path / "work3/s234805"
    run_storage = storage_dir / "codllm"
    for path in (
        run_storage / "cache/custom",
        run_storage / ".venv/lib/python3.13/site-packages/pkg",
        run_storage / "python/cpython",
    ):
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(b"cache")

    cfg = Config(
        data_processed_dir=str(run_storage / "data/processed"),
        output_dir=str(run_storage / "runs"),
        processed_filename="data.parquet",
    )

    actions = clear_generated_caches(
        cfg,
        repo_dir=tmp_path,
        environ={
            "STORAGE_FOLDER": str(storage_dir),
            "RUN_STORAGE_DIR": str(run_storage),
        },
        mode="aggressive",
        include_env_caches=True,
        execute=True,
    )

    assert any(action.entry.path == run_storage / "cache" for action in actions)
    assert any(action.entry.path == run_storage / ".venv" for action in actions)
    assert any(action.entry.path == run_storage / "python" for action in actions)
    assert not (run_storage / "cache").exists()
    assert not (run_storage / ".venv").exists()
    assert not (run_storage / "python").exists()


def test_maintenance_status_reports_capacity_and_known_categories(
    tmp_path: Path,
) -> None:
    """Maintenance status should summarize disk roots and known storage buckets."""
    _git(tmp_path, "init")
    tracked_file = tmp_path / "README.md"
    tracked_file.write_text("tracked\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    raw_dir = tmp_path / "data/raw"
    processed_dir = tmp_path / "data/processed"
    run_dir = tmp_path / "runs/run-0001"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (raw_dir / "source.csv").write_text("raw\n", encoding="utf-8")
    (processed_dir / "data.parquet").write_bytes(b"processed")
    (run_dir / "model.safetensors").write_bytes(b"weights")
    cfg = Config(
        data_raw_dir=str(raw_dir),
        data_processed_dir=str(processed_dir),
        output_dir=str(tmp_path / "runs"),
        processed_filename="data.parquet",
    )

    report = build_maintenance_status(cfg, repo_dir=tmp_path, environ={})
    categories = {category.name: category.size_bytes for category in report.categories}

    assert report.roots
    assert categories["code and tracked specs"] == len("tracked\n")
    assert categories["raw datasets"] == len("raw\n")
    assert categories["processed datasets and split caches"] == len(b"processed")
    assert categories["model weights and run outputs"] == len(b"weights")


def test_maintenance_status_reports_uncategorized_hpc_storage(
    tmp_path: Path,
) -> None:
    """Maintenance status should expose managed and uncategorized HPC storage."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git(repo_dir, "init")
    storage_dir = tmp_path / "work3/s234805"
    run_storage = storage_dir / "codllm"
    cache_dir = run_storage / "cache/custom"
    cache_dir.mkdir(parents=True)
    unknown_path = run_storage / "unknown.bin"
    sibling_dir = storage_dir / "other-project"
    sibling_dir.mkdir(parents=True)
    (cache_dir / "payload.bin").write_bytes(b"known-cache")
    unknown_path.write_bytes(b"unknown")
    (sibling_dir / "payload.bin").write_bytes(b"sibling")
    cfg = Config(
        data_raw_dir=str(repo_dir / "data/raw"),
        data_processed_dir=str(run_storage / "data/processed"),
        output_dir=str(run_storage / "runs"),
        processed_filename="data.parquet",
    )

    report = build_maintenance_status(
        cfg,
        repo_dir=repo_dir,
        environ={
            "STORAGE_FOLDER": str(storage_dir),
            "RUN_STORAGE_DIR": str(run_storage),
        },
    )
    categories = {category.name: category.size_bytes for category in report.categories}

    assert categories["runtime dependency caches"] == len(b"known-cache")
    assert categories["HPC run storage total"] == len(b"known-cache") + len(b"unknown")
    assert categories["uncategorized HPC run storage"] == len(b"unknown")
    assert categories["HPC run storage child: cache"] == len(b"known-cache")
    assert categories["HPC run storage child: unknown.bin"] == len(b"unknown")
    assert categories["HPC storage folder total"] == (
        len(b"known-cache") + len(b"unknown") + len(b"sibling")
    )
    assert categories["HPC storage sibling: other-project"] == len(b"sibling")


def test_maintenance_status_reports_quota_gap_for_storage_folder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Maintenance status should flag quota usage not visible below STORAGE_FOLDER."""
    _git(tmp_path, "init")
    original_run = status_module.subprocess.run
    original_which = status_module.shutil.which
    storage_dir = tmp_path / "work3/s234805"
    run_storage = storage_dir / "codllm"
    run_storage.mkdir(parents=True)
    (run_storage / "visible.bin").write_bytes(b"visible")

    def fake_which(command: str) -> str | None:
        if command == "getquota_work3.sh":
            return f"/usr/bin/{command}"
        return original_which(command)

    def fake_run(command, **kwargs):
        if command == ["getquota_work3.sh"]:
            return SimpleNamespace(
                returncode=0,
                stdout="s234805 |54321 || 10.00 GiB| 300.00 GiB|| 119758 | 2000000\n",
                stderr="",
            )
        return original_run(command, **kwargs)

    monkeypatch.setattr(status_module.shutil, "which", fake_which)
    monkeypatch.setattr(status_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        status_module,
        "_quota_command_for_path",
        lambda path: ("getquota_work3.sh",)
        if path.resolve(strict=False) == storage_dir.resolve(strict=False)
        else None,
    )

    cfg = Config(
        data_processed_dir=str(run_storage / "data/processed"),
        output_dir=str(run_storage / "runs"),
    )

    report = build_maintenance_status(
        cfg,
        repo_dir=tmp_path,
        environ={
            "STORAGE_FOLDER": str(storage_dir),
            "RUN_STORAGE_DIR": str(run_storage),
        },
    )
    categories = {category.name: category.size_bytes for category in report.categories}

    assert categories["quota not visible under configured HPC storage folder"] == (
        10 * 1024**3 - len(b"visible")
    )


def test_quota_parser_handles_dtu_zhome_and_work3_outputs() -> None:
    """Quota parsing should handle DTU home and scratch quota command formats."""
    zhome = _parse_quota_output("You are using 12.34 GB of 30.00 GB.")
    work3 = _parse_quota_output(
        """
          user/group     ||           size          ||    chunk files
             name |  id  ||    used    |    hard    ||  used   |  hard
        --------------|------||------------|------------||---------|---------
          s123456 |54321 ||  158.75 GiB|  400.00 GiB||  119758 |  2000000
        """
    )

    assert zhome == (12_340_000_000, 30_000_000_000)
    assert work3 == (int(158.75 * 1024**3), 400 * 1024**3)


def test_maintenance_status_attaches_dtu_work3_quota(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Maintenance status should attach DTU quota data for /work3 roots."""
    _git(tmp_path, "init")
    original_run = status_module.subprocess.run
    original_which = status_module.shutil.which

    def fake_which(command: str) -> str | None:
        if command == "getquota_work3.sh":
            return f"/usr/bin/{command}"
        return original_which(command)

    def fake_run(command, **kwargs):
        if command == ["getquota_work3.sh"]:
            return SimpleNamespace(
                returncode=0,
                stdout="s234805 |54321 || 390.00 GiB| 400.00 GiB|| 119758 | 2000000\n",
                stderr="",
            )
        return original_run(command, **kwargs)

    monkeypatch.setattr(status_module.shutil, "which", fake_which)
    monkeypatch.setattr(status_module.subprocess, "run", fake_run)

    cfg = Config(
        data_processed_dir=str(tmp_path / "data/processed"),
        output_dir=str(tmp_path / "runs"),
    )

    report = build_maintenance_status(
        cfg,
        repo_dir=tmp_path,
        environ={"RUN_STORAGE_DIR": "/work3/s234805/codllm"},
    )

    run_storage = next(root for root in report.roots if root.name == "HPC run storage")

    assert run_storage.path == Path("/work3/s234805/codllm")
    assert run_storage.quota is not None
    assert run_storage.quota.source == "getquota_work3.sh"
    assert run_storage.quota.used_bytes == 390 * 1024**3
    assert run_storage.quota.limit_bytes == 400 * 1024**3
    assert run_storage.quota.free_bytes == 10 * 1024**3


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


def _touch_tree(path: Path, timestamp: float) -> None:
    """Set mtime/atime recursively for one test path."""
    for child in path.rglob("*"):
        os.utime(child, (timestamp, timestamp))
    os.utime(path, (timestamp, timestamp))
