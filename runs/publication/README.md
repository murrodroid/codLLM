# Publication experiments

This directory contains the first publication-optimization stage: three
controlled response curves and a native W&B Bayesian interaction search. The
full research design and promotion gates are in
[`publication_plan.md`](../../publication_plan.md).

## Frozen discovery protocol

All response curves inherit `base.toml`. It explicitly pins the model, data
contract, scheduler, augmentation, checkpoint selection, and both random seeds:

```text
CODLLM_SEED=777
CODLLM_DATA_SEED=777
CODLLM_WANDB_MODE=online
CODLLM_WANDB_LOG_MODEL=false
```

The response curves are conditional on the latest model recipe: five-label
seq2seq prediction, COD+age+sex inputs, floor balancing, synthetic multi-COD,
base perturbation, and masterlist pretraining. Fine-tuning uses an explicitly
pinned cosine scheduler. The subsequent Bayesian search compares cosine with
constant-with-warmup behavior.

Test and uncertainty evaluation are disabled during tuning with:

```text
CODLLM_FINAL_TEST_EVAL_ENABLED=false
CODLLM_UNCERTAINTY_EVAL=false
```

The test split is still constructed and remains fixed. Only its evaluation is
suppressed. A later, explicitly declared finalist specification must enable it.

Pretraining early stopping and best-checkpoint reloading are independently
disabled, so the pretraining-dose variants reach the declared pretraining epoch
before fine-tuning:

```text
CODLLM_PRETRAIN_EARLY_STOPPING_PATIENCE=0
CODLLM_PRETRAIN_LOAD_BEST_MODEL_AT_END=false
```

Fine-tuning retains patience 10 and reloads its best macro-F1 checkpoint.

## Response curves

| File | Grid | Runs |
|---|---|---:|
| `balance_floor_curve.toml` | 0, 100, 200, 300, 450, 600, 900 | 7 |
| `multicod_synthetic_curve.toml` | 0, .15, .30, .45, .60, .80, 1.0 | 7 |
| `pretraining_dose_curve.toml` | off, 4, 8, 16, 32, 48 epochs | 6 |

Each curve uses lockstep variants and gives every cell a distinct checkpoint
root. This is required for safe parallel execution and multi-slot auto-resume.
Each scheduler slot reuses a W&B run-id sidecar in its stable run-state
directory. High-level progress is written directly to the W&B Logs tab, and
intermediate slots skip final-model artifact upload so live telemetry can flush
before resubmission.

On the HPC login node:

```bash
cd ~/codLLM
source hpc/env.sh

uv run --no-sync invoke experiments.plan \
  --config runs/publication/balance_floor_curve.toml

uv run --no-sync invoke hpc.build \
  --config runs/publication/balance_floor_curve.toml \
  --profile h100 \
  --lucas

uv run --no-sync invoke hpc.submit \
  --config runs/publication/balance_floor_curve.toml \
  --profile h100 \
  --lucas \
  --duration 2w
```

Repeat those commands for `multicod_synthetic_curve.toml` and
`pretraining_dose_curve.toml`. Inspect `experiments.plan` and storage before
submitting. A two-week campaign is an upper budget, not a requirement; early
stopping and completion markers end cells earlier.

## Bayesian search

`bayesian_sweep.yaml` defines a 60-trial W&B Bayesian search. Its child program
loads `bayesian_base.toml`, validates an explicit parameter allowlist, maps the
sample into existing `CODLLM_*` settings, and gives it a trial-specific output
root.

The objective is logged once after genuine completion:

```text
optimization/val_macro_f1
```

Bayesian trials are medium-fidelity screening runs with 24 fine-tuning epochs,
no final test evaluation, no model artifact upload, and no codLLM auto-resume.
Every trial must fit one 24-hour H100 allocation.

Create the sweep:

```bash
cd ~/codLLM
source hpc/env.sh

uv run --no-sync invoke experiments.bayes-create \
  --config runs/publication/bayesian_sweep.yaml \
  --project codllm \
  --entity murromanden_data
```

Copy the real `entity/project/sweep-id` printed by W&B. Submit a wave of four
single-trial agents:

```bash
uv run --no-sync invoke hpc.bayes-submit \
  --sweep-id murromanden_data/codllm/<sweep-id> \
  --agents 4 \
  --profile h100 \
  --lucas
```

Submit further waves after earlier trials finish so Bayesian assignment benefits
from their results. Each array element runs exactly:

```text
wandb agent --forward-signals --count 1 <sweep-id>
```

Do not add `--duration`, do not run more than one trial in an allocation, and do
not enable Hyperband yet. The current training lifecycle intentionally supports
ordinary completion and wall-time resume, but it does not yet have a distinct
pruned-run state.

Preview an agent submission without calling `bsub`:

```bash
uv run --no-sync invoke hpc.bayes-submit \
  --sweep-id murromanden_data/codllm/<sweep-id> \
  --agents 4 \
  --profile h100 \
  --lucas \
  --dry-run
```

## Analysis and promotion

Use validation macro F1 for selection. Retain exact accuracy, unseen-string macro
F1, per-source metrics, single/multi-COD metrics, generated rows, and runtime for
Pareto analysis.

For each response curve, predeclare saturation as two consecutive increments
below 0.002 macro F1 whose paired 95% confidence-interval upper bound is below
0.005. Repeat the plateau neighborhood across at least three seeds.

Promote the five best Bayesian candidates into ordinary TOML variants for
full-fidelity 100–120 epoch, multi-slot training. Bayesian agent jobs themselves
must never be resumed across scheduler allocations.
