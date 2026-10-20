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

## Interactive A100 analysis

After establishing the shared connection, open an interactive login shell:

```shell
ssh -S /tmp/codex-dtu-hpc.sock -o ControlMaster=no -o ProxyCommand=false -tt s234805@login.hpc.dtu.dk
a100sh
```

Run `a100sh` on the HPC login node. It submits an interactive `gpua100i` job and waits for dispatch
before opening an A100 shell. Record the printed LSF job ID. On the allocated node:

```shell
cd ~/codLLM
source hpc/env.sh
```

Use `uv run --no-sync` for analysis and checks after dependencies have been synced. Respect the
assigned devices and shared-node resources; batch queues remain the normal route for sustained training.
Do not overwrite unrelated changes in the HPC checkout when transferring analysis code.

Always stop your analysis processes and `exit` the A100 shell when finished. Back on the login node,
check `bjobs -a <interactive-job-id>` to verify that the allocation has ended, then `exit` your SSH
client. Do not leave a floating interactive session. Closing this client is separate from
`connect --disconnect`, which closes the shared master and affects other clients.
Scheduler accounting can briefly show `RUN` after the node shell exits; check again until it reports
`DONE`/`EXIT` or is no longer active. A noninteractive SSH check may need
`bash -lc 'bjobs -a <interactive-job-id>'` to load the scheduler command path.
