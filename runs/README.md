# Run specifications

Every experiment is a declarative TOML spec. Specs compose by inheritance: a spec's
`base` field pulls in one or more parent specs (resolved relative to the spec's own
directory) and the merged environment is what runs. Run outputs (`run-*/`, checkpoints)
are gitignored; only the specs are tracked, so a clone gives ready-to-submit recipes.

## Tiers

| Directory | Role |
|---|---|
| `profiles/` | Per-model **runtime** profiles (model name, batch size, dataloader): `h100-{small,base,large,xl}.toml`. Distinct from `hpc/lsf_profiles.toml`, which sets the LSF **submission** resources (queue, wall time, cores, memory, GPU). |
| `sweeps/` | Reusable **sweep building-blocks**: one factor varied over a few values (multicod ratio, LR scheduler, perturbation, pretraining, training inputs). Each inherits a runtime profile. |
| `publication/` | Publication-grade response curves and W&B Bayesian screening. See `publication/README.md` for test-leakage, output-isolation, and single-slot-agent safeguards. |
| `thesis/` | The **reported thesis experiments**. Each composes a sweep building-block with the locked thesis base config. `model_size.toml` is the size sweep (small to xl); the others are the one-factor ablations. |
| `single/` | **Standalone runs**: the five leave-one-source-out holdouts (`base_holdout_*`, RQ3), the deployment recipe (`size_sweep_base_final.toml`), and individual model runs (`codllm_*`). |

## Inheritance example

```
runs/thesis/scheduler.toml
  base = ["../sweeps/scheduler.toml",   # the LR-scheduler sweep grid
          "base_thesis.toml"]           # the locked thesis base config
              |- runs/sweeps/scheduler.toml   -> base = "../profiles/h100-large.toml"
              |- runs/thesis/base_thesis.toml -> base = "../profiles/h100-base.toml"
```

## Running a spec

```
# preview the expanded runs without submitting
uv run invoke experiments.plan --config runs/thesis/model_size.toml

# submit to the HPC (the --profile here is the LSF submission profile, not runs/profiles/)
uv run invoke hpc.submit --config runs/thesis/model_size.toml --profile h100 --user <name>
```

See the [project guide](../docs/guide.md) for the full reproduce table and HPC bootstrap.
