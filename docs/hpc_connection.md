# Connect from your laptop

Run from the repository root in your **local terminal**, not inside ThinLinc/HPC:

```shell
uv run connect --lucas
```

Enter any password or key passphrase directly into SSH's terminal prompt. The helper uses your
existing key and `authorized_keys` setup; it does not create keys or store credentials. Host-key
verification remains enabled. Settings are in `hpc/ssh_profiles.toml`; never put secrets there.

The command creates a background SSH master, or reuses a responsive existing connection. On macOS
the configured `/tmp/codex-dtu-hpc.sock` resolves to `/private/tmp/codex-dtu-hpc.sock`, preserving the
existing workflow. The session persists for up to eight idle hours; sleep or network changes can
interrupt it. It does not open an interactive shell or submit jobs.

Check connectivity without logging in:

```shell
uv run connect --lucas --status
```

Explicitly close the shared connection (also interrupts clients using it):

```shell
uv run connect --lucas --disconnect
```

Preview without connecting:

```shell
uv run connect --lucas --dry-run
```

After the first `uv run` installs the new entrypoint, `uv run --no-sync connect --lucas` also works.
Other profiles can be selected with `--user <name>` after adding their settings to the TOML file.
Do not delete an unfamiliar or unresponsive socket blindly; the helper deliberately does not remove it.
