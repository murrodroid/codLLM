"""Git maintenance helpers for generated logs and push hygiene."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess

GENERATED_PATH_PROBES: tuple[str, ...] = (
    "logs/_codllm_maintenance_probe_",
    "jobs/generated/_codllm_maintenance_probe_",
    "runs/run-0001/_codllm_maintenance_probe_",
    "runs_smoke/_codllm_maintenance_probe_",
    "data/cache/_codllm_maintenance_probe_",
    "data/processed/data.parquet",
    "data/processed/data.splits/_codllm_maintenance_probe_",
)

GENERATED_STATUS_PATHS: tuple[str, ...] = (
    "logs",
    "jobs/generated",
    "runs/run-0001",
    "runs_smoke",
    "data/cache",
    "data/processed",
)


@dataclass(frozen=True)
class GitHygieneIssue:
    """One git hygiene issue for generated or local-only files."""

    path: str
    message: str


@dataclass(frozen=True)
class GitHygieneReport:
    """Git hygiene check results for generated paths."""

    repo_dir: Path
    ignored_probes: dict[str, bool]
    status_entries: tuple[str, ...]
    issues: tuple[GitHygieneIssue, ...]

    @property
    def has_issues(self) -> bool:
        """Return True when generated paths may leak into git pushes."""
        return bool(self.issues)


def build_git_hygiene_report(
    repo_dir: Path | str = ".",
    *,
    generated_probes: tuple[str, ...] = GENERATED_PATH_PROBES,
    status_paths: tuple[str, ...] = GENERATED_STATUS_PATHS,
) -> GitHygieneReport:
    """Check that generated output probes are ignored and unstaged."""
    repo_path = Path(repo_dir)
    ignored_probes: dict[str, bool] = {}
    issues: list[GitHygieneIssue] = []
    for probe in generated_probes:
        ignored = _git(repo_path, ["check-ignore", "-q", "--", probe]).returncode == 0
        ignored_probes[probe] = ignored
        if not ignored:
            issues.append(
                GitHygieneIssue(
                    path=probe,
                    message="Generated-path probe is not ignored by git.",
                )
            )

    status_entries = _git_status_entries(repo_path, status_paths)
    for entry in status_entries:
        issues.append(
            GitHygieneIssue(
                path=entry,
                message="Generated/local-only path appears in git status.",
            )
        )

    return GitHygieneReport(
        repo_dir=repo_path,
        ignored_probes=ignored_probes,
        status_entries=tuple(status_entries),
        issues=tuple(issues),
    )


def write_git_snapshot(
    repo_dir: Path | str = ".",
    *,
    output_dir: Path | str = "logs/git",
    max_commits: int = 20,
) -> Path:
    """Write a git status and recent-commit snapshot to an ignored log directory."""
    repo_path = Path(repo_dir)
    output_path = repo_path / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_path = output_path / f"git-snapshot-{timestamp}.txt"
    sections = [
        ("timestamp", datetime.now().isoformat(timespec="seconds")),
        ("repo", _git_text(repo_path, ["rev-parse", "--show-toplevel"])),
        ("branch", _git_text(repo_path, ["branch", "--show-current"])),
        ("head", _git_text(repo_path, ["rev-parse", "HEAD"])),
        ("status", _git_text(repo_path, ["status", "--short"])),
        (
            "recent commits",
            _git_text(repo_path, ["log", "--oneline", f"-n{max(1, max_commits)}"]),
        ),
    ]
    content = "\n\n".join(
        f"## {title}\n{body or '<empty>'}" for title, body in sections
    )
    snapshot_path.write_text(f"{content}\n", encoding="utf-8")
    return snapshot_path


def _git_status_entries(repo_dir: Path, paths: tuple[str, ...]) -> list[str]:
    """Return porcelain git status lines for selected paths."""
    entries: list[str] = []
    for path in paths:
        result = _git(repo_dir, ["status", "--short", "--", path])
        if result.stdout.strip():
            entries.extend(result.stdout.strip().splitlines())
    return entries


def _git_text(repo_dir: Path, args: list[str]) -> str:
    """Return git command stdout, or a compact failure message."""
    result = _git(repo_dir, args)
    if result.returncode == 0:
        return result.stdout.strip()
    return f"<git {' '.join(args)} failed: {result.stderr.strip()}>"


def _git(repo_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command in text mode without raising on failure."""
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        text=True,
        capture_output=True,
        check=False,
    )
