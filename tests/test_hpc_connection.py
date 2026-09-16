"""CPU-only tests of the laptop SSH helper; never contact a real server."""

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from codllm import hpc_connection


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    """Create a temporary profile and dummy identity without using real credentials."""
    identity = tmp_path / "key"
    identity.touch()
    path = tmp_path / "profiles.toml"
    path.write_text(
        f'[users.lucas]\naccount = "test-user"\nidentity = "{identity}"\nsocket = "{tmp_path / "socket"}"\n'
    )
    return path


def test_dry_run_never_calls_ssh(profile, monkeypatch, capsys):
    """A preview preserves host verification and avoids credential/agent forwarding."""
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    assert (
        hpc_connection.main(["--lucas", "--profiles", str(profile), "--dry-run"]) == 0
    )
    command = capsys.readouterr().out
    assert "ControlPersist=8h" in command
    assert "ForwardAgent=no" in command
    assert "StrictHostKeyChecking=no" not in command
    run.assert_not_called()


@pytest.mark.parametrize("connected", [False, True])
def test_status_only_probes(profile, monkeypatch, connected):
    """Status cannot create a fresh authenticated connection."""
    run = Mock(return_value=subprocess.CompletedProcess([], 0 if connected else 255))
    monkeypatch.setattr(subprocess, "run", run)
    assert hpc_connection.main(["--lucas", "--profiles", str(profile), "--status"]) == (
        0 if connected else 1
    )
    assert run.call_count == (2 if connected else 1)
    assert all("-Nf" not in call.args[0] for call in run.call_args_list)
    if connected:
        assert "ProxyCommand=false" in run.call_args.args[0]
        assert "BatchMode=yes" in run.call_args.args[0]


def test_connect_then_verify(profile, monkeypatch):
    """Authentication inherits the terminal; success requires a responsive remote."""
    run = Mock(
        side_effect=[subprocess.CompletedProcess([], code) for code in (255, 0, 0)]
    )
    monkeypatch.setattr(subprocess, "run", run)
    assert hpc_connection.main(["--lucas", "--profiles", str(profile)]) == 0
    auth = run.call_args_list[1]
    assert "-Nf" in auth.args[0]
    assert "capture_output" not in auth.kwargs
    assert "input" not in auth.kwargs


@pytest.mark.parametrize("codes", [(0, 0), (0, 255), (255, 255), (255, 0, 255)])
def test_existing_or_failed_connection(profile, monkeypatch, codes):
    """Reuse healthy sessions, preserve broken masters, and propagate failures."""
    run = Mock(side_effect=[subprocess.CompletedProcess([], code) for code in codes])
    monkeypatch.setattr(subprocess, "run", run)
    result = hpc_connection.main(["--lucas", "--profiles", str(profile)])
    assert (result == 0) == (codes == (0, 0))
    assert run.call_count == len(codes)


def test_stale_socket_is_not_removed(profile, monkeypatch):
    """Never overwrite an unexpected socket path automatically."""
    socket = profile.parent / "socket"
    socket.write_text("preserve me")
    run = Mock(return_value=subprocess.CompletedProcess([], 255))
    monkeypatch.setattr(subprocess, "run", run)
    assert hpc_connection.main(["--lucas", "--profiles", str(profile)]) == 1
    assert socket.read_text() == "preserve me"
    assert run.call_count == 1


def test_missing_key(profile, monkeypatch):
    """Missing credentials produce an actionable failure without starting authentication."""
    (profile.parent / "key").unlink()
    run = Mock(return_value=subprocess.CompletedProcess([], 255))
    monkeypatch.setattr(subprocess, "run", run)
    assert hpc_connection.main(["--lucas", "--profiles", str(profile)]) == 1
    assert run.call_count == 1


def test_disconnect_is_explicit(profile, monkeypatch):
    """The exit control command is only issued for an explicit disconnect request."""
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    assert (
        hpc_connection.main(["--lucas", "--profiles", str(profile), "--disconnect"])
        == 0
    )
    assert "exit" in run.call_args.args[0]


def test_probe_timeout(profile, monkeypatch):
    """An unresponsive master yields disconnected status without blocking indefinitely."""
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("ssh", 20))
    )
    assert hpc_connection.main(["--lucas", "--profiles", str(profile), "--status"]) == 1


def test_unknown_profile(profile):
    """Unknown aliases fail during argument validation."""
    with pytest.raises(SystemExit) as exc:
        hpc_connection.main(["--user", "missing", "--profiles", str(profile)])
    assert exc.value.code == 2
