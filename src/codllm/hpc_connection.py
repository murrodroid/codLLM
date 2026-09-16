"""Manage a laptop-side shared SSH connection without handling credentials."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """User-editable connection defaults, separate from model runtime settings."""

    account: str
    identity: str
    socket: str
    host: str = "login.hpc.dtu.dk"
    persist: str = "8h"
    connect_timeout: int = 15
    probe_timeout: int = 20
    keepalive_interval: int = 60
    keepalive_count: int = 3

    @property
    def target(self) -> str:
        """Return the SSH destination without invoking a shell."""
        return f"{self.account}@{self.host}"

    @property
    def socket_path(self) -> str:
        """Resolve /tmp to /private/tmp on macOS for existing-session compatibility."""
        return str(Path(self.socket).expanduser().resolve())

    def control_command(self, operation: str) -> list[str]:
        """Build a local SSH control command that cannot initiate authentication."""
        return ["ssh", "-S", self.socket_path, "-O", operation, self.target]

    def probe_command(self) -> list[str]:
        """Check remote responsiveness using only the existing master connection."""
        return [
            "ssh",
            "-S",
            self.socket_path,
            "-o",
            "ControlMaster=no",
            "-o",
            "ProxyCommand=false",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            self.target,
            "true",
        ]

    def connect_command(self) -> list[str]:
        """Build a persistent connection with interactive SSH authentication."""
        return [
            "ssh",
            "-i",
            str(Path(self.identity).expanduser()),
            "-o",
            "IdentitiesOnly=yes",
            "-M",
            "-S",
            self.socket_path,
            "-o",
            f"ControlPersist={self.persist}",
            "-o",
            f"ServerAliveInterval={self.keepalive_interval}",
            "-o",
            f"ServerAliveCountMax={self.keepalive_count}",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ForwardX11=no",
            "-o",
            "ClearAllForwardings=yes",
            "-Nf",
            self.target,
        ]


def _succeeds(command: list[str], timeout: int) -> bool:
    """Run a bounded, noninteractive diagnostic without noisy expected failures."""
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Connect, check, or explicitly close the selected user's shared SSH session."""
    parser = argparse.ArgumentParser(description=__doc__)
    users = parser.add_mutually_exclusive_group(required=True)
    users.add_argument("--lucas", action="store_true", help="Use the Lucas profile")
    users.add_argument("--user", help="Select a user in hpc/ssh_profiles.toml")
    parser.add_argument("--profiles", type=Path, default=Path("hpc/ssh_profiles.toml"))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--status",
        action="store_true",
        help="Check connectivity without authenticating",
    )
    actions.add_argument(
        "--disconnect", action="store_true", help="Close the shared connection"
    )
    actions.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the connection command without running SSH",
    )
    args = parser.parse_args(argv)
    user = "lucas" if args.lucas else args.user
    try:
        profiles = tomllib.loads(args.profiles.read_text())
        cfg = Config(**profiles["users"][user])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"Cannot load SSH profile {user!r} from {args.profiles}: {exc}")
    if args.dry_run:
        print(shlex.join(cfg.connect_command()))
        return 0
    try:
        if args.disconnect:
            return subprocess.run(
                cfg.control_command("exit"), check=False, timeout=cfg.probe_timeout
            ).returncode
        master_alive = _succeeds(cfg.control_command("check"), cfg.probe_timeout)
        connected = master_alive and _succeeds(cfg.probe_command(), cfg.probe_timeout)
        if connected:
            print(f"Connected: {cfg.target} (socket: {cfg.socket_path})")
            return 0
        if args.status:
            print(f"Not connected: {cfg.target}. Run uv run connect --user {user}")
            return 1
        if master_alive:
            print(
                "The SSH master exists but HPC is not responding. Check your network/VPN, or explicitly"
            )
            print(
                f"close it with uv run connect --user {user} --disconnect before reconnecting."
            )
            return 1
        if Path(cfg.socket_path).exists() or Path(cfg.socket_path).is_symlink():
            print(
                f"Unresponsive socket exists: {cfg.socket_path}. Refusing to overwrite it; inspect it first."
            )
            return 1
        if not Path(cfg.identity).expanduser().is_file():
            print(
                f"SSH key not found: {cfg.identity}. Configure an existing key in {args.profiles}."
            )
            return 1
        print(
            f"Connecting to {cfg.target}. Enter any SSH password/passphrase in your terminal.",
            flush=True,
        )
        result = subprocess.run(cfg.connect_command(), check=False)
        if result.returncode:
            return result.returncode
        if not _succeeds(cfg.probe_command(), cfg.probe_timeout):
            print(
                "SSH started, but the remote connectivity check failed. Check your network/VPN and retry --status."
            )
            return 1
        print(
            f"Connected: {cfg.target} (socket: {cfg.socket_path}, idle persistence: {cfg.persist})"
        )
        return 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"SSH failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
