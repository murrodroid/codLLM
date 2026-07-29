# codLLM: Coding historical causes of death with language models

![python](https://img.shields.io/badge/python-3.13-blue) ![license](https://img.shields.io/badge/license-MIT-green)

**Thesis:** [codLLM_Thesis.pdf](codLLM_Thesis.pdf)

## Abstract

Historical mortality registers record each death as a free-text cause, written in the language and orthography of its
time and place. Before such data can be analyzed, every entry must be mapped by hand to a standardized historical
disease code (ICD10h), expert work slow enough to gate whole research projects. This thesis asks whether a fine-tuned,
task-specific language model can take over the bulk of that coding within a human-in-the-loop workflow, and whether the
confidence of its predictions can make it safe to deploy.

We fine-tune the FLAN-T5 sequence-to-sequence model at four sizes on roughly 1.5 million hand-coded records from five
European registers spanning Danish, Dutch, English, French, and Spanish, and evaluate the resulting deployment model,
`codLLM-base-deploy`, on a locked test split against classical, retrieval-based, and agentic baselines, under
uncertainty-based selective prediction, and under transfer to an entirely held-out source. On the full test set it
reaches 0.985 exact-match accuracy, 0.987 micro-F1, and 0.749 macro-F1, against 0.314 macro-F1 for the strongest
training-free baseline, a TF-IDF classifier, and it far exceeds two frontier agents equipped with tools (0.56 to 0.60
accuracy on a matched subset) at a fraction of their per-record cost. On the memorization-free unseen-string slice,
where the model is weakest, it still reaches 0.843 exact-match accuracy and 0.625 macro-F1, so its advantage reflects
genuine generalization rather than recall of seen strings.

Reading the model's own token-level confidence makes the workflow usable. Auto-accepting the most-confident 80% of
predictions codes them at near-perfect accuracy (above 0.999) while routing the uncertain remainder to a historical
demographer, and on the unseen slice the same signal lifts retained accuracy from 0.843 to 0.925, deferring the
genuinely novel inputs rather than coding them confidently wrong. Transfer to a source the model never trained on is the
hardest setting: exact-match accuracy on a held-out source ranges from 0.22 to 0.64 and macro-F1 from 0.09 to 0.36, and
although confidence stays informative, no operating point reaches an accuracy that would license unsupervised coding, so
on a genuinely new register the model is a decision-support tool rather than an autonomous coder. We treat the dataset's
heavy duplication explicitly throughout, reporting a memorization-free slice and duplication-aware confidence intervals,
and recommend the compute-light base model for deployment.

![codLLM training pipeline](visualizations/codLLM_pipeline.png)

## Problem

Historical death registers record causes of death as free text, in several languages, with archaic and inconsistent
terminology. Turning that text into standardized ICD10h codes is normally manual, slow, and dependent on specialist
knowledge. codLLM standardizes five European registers (about 1.5M records), harmonizes their labels to the 2024 ICD10h
masterlist, and fine-tunes sequence-to-sequence models to do the coding automatically, with uncertainty estimates that
tell a historian which records are safe to accept and which to review. The historical datasets are not public and must
be supplied separately under `data/raw/`.

## Research questions

> **RQ1.** On historical, multilingual ICD10h cause-of-death coding, does our fine-tuned approach outperform classical,
> retrieval-based, and agentic-LLM baselines, on both the full test set and the unseen-string slice?
>
> **RQ2.** Can uncertainty-informed selective prediction keep retained predictions highly accurate while deferring
> ambiguous cases to expert review?
>
> **RQ3.** Can the approach reliably classify records from a historical source it was never trained on, and where it
> cannot, does its uncertainty track its accuracy so those records can be deferred to expert review?

RQ1 is answered by `experiments/baselines/` and `experiments/agentic_baseline_v2/` against the fine-tuned size sweep;
RQ2 and RQ3 by `experiments/uncertainty/` and the leave-one-source-out specs under `runs/`.

## Repository map

```text
codLLM/
  src/codllm/            # the package: training, inference, data, settings, experiments
  runs/                  # declarative TOML experiment specs  (see runs/README.md)
    profiles/            #   per-model runtime profiles (h100-{small,base,large,xl})
    sweeps/              #   reusable single-factor sweep building-blocks
    thesis/              #   the reported thesis experiments (size sweep + ablations)
    single/             #   standalone runs: holdouts (RQ3), deployment recipe
  experiments/           # research code  (see experiments/README.md)
    baselines/           #   RQ1 classical baselines
    agentic_baseline_v2/ #   RQ1 agentic baseline
    offline_test_eval/   #   authoritative held-out test evaluation
    uncertainty/         #   RQ2 selective prediction + RQ3 transfer
    notebooks/           #   exploratory data analysis
    exploratory/         #   superseded directions (RAG, agentic v1)
  hpc/                   # LSF submission profiles + storage bootstrap
  tests/                 # pytest suite
  docs/                  # onboarding
  data/                  # raw + processed data (gitignored; supplied separately)
```

## Reproduce every experiment

Training runs are submitted with `uv run invoke hpc.submit --config <spec> --profile <lsf-profile> --user <name>`.
The W&B `test/*` metric is aliased to validation by a known trainer bug, so the **authoritative held-out test number
comes from `experiments/offline_test_eval/eval.py`**, not from W&B.

| Experiment | Spec or command |
|---|---|
| Size sweep, small to xl (RQ1) | `runs/thesis/model_size.toml` |
| One-factor ablations (floor, multicod, perturbation, scheduler, inputs, pretraining, masterlist) | `runs/thesis/{upsampling_floor,multicod_synthetic,perturbation_rate,scheduler,input_features,pretraining,masterlist_injection}.toml` |
| Deployment recipe (ablation winners) | `runs/single/size_sweep_base_final.toml` |
| Leave-one-source-out holdouts (RQ3) | `runs/single/base_holdout_{amsterdam,belgium,copenhagen,ipswich,madrid}.toml` |
| Authoritative held-out test eval | `uv run python -m experiments.offline_test_eval.eval --checkpoint <ckpt> --out <json>` |
| Classical baselines (RQ1) | `uv run python -m experiments.baselines.{tfidf,embedding,majority_class}_baseline` |
| Agentic baseline (RQ1) | `uv run python -m experiments.agentic_baseline_v2.run` then `...eval_posthoc` |
| Selective prediction (RQ2) | `experiments/uncertainty/{dump_test_split,local_infer,analysis}.py` |
| Transfer + uncertainty (RQ3) | `experiments/uncertainty/_rq3_*.py` |
| Local smoke run | `uv run invoke train --config runs/single/codllm_small.toml` (override sizes, see Quick Start) |

See [`runs/README.md`](runs/README.md) for spec inheritance and [`experiments/README.md`](experiments/README.md) for
the research layout.

## Quick Start

Requirements:

- Python 3.13
- `uv`
- raw dataset files in `data/raw/` for training
- a Hugging Face token for private models or rate-limited downloads
- an optional W&B API key for experiment tracking

Install the project:

```bash
git clone https://github.com/murrodroid/codllm
cd codllm
uv sync
```

Set credentials permanently in your shell environment:

```zsh
SHELL_RC="$HOME/.profile"
case "$(basename "${SHELL:-}")" in
  zsh) SHELL_RC="${ZDOTDIR:-$HOME}/.zshrc" ;;
  bash) SHELL_RC="$HOME/.bashrc" ;;
esac

cat <<'EOF' >> "$SHELL_RC"

export HF_TOKEN="..."
export WANDB_API_KEY="..."
EOF

source "$SHELL_RC"
```

Run tests:

```bash
uv run pytest tests/
```

Run local training directly from environment/config defaults:

```bash
uv run python -m codllm.training
```

Run one TOML experiment spec locally:

```bash
uv run invoke train --config runs/single/codllm_small.toml
```

The checked-in specs target the project workflow, and several inherit H100 batch-size defaults. Adjust batch sizes,
dataset size, and epochs before running them on smaller local hardware.

For a local smoke run, override the expensive settings in your shell before training, for example:

```bash
export CODLLM_DATASET_SIZE=0.01
export CODLLM_NUM_TRAIN_EPOCHS=1
export CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE=4
export CODLLM_PER_DEVICE_EVAL_BATCH_SIZE=4
export CODLLM_WANDB_MODE=disabled
uv run python -m codllm.training
```

## Data Inputs

Default source configuration lives in `Config.data_sources` in `src/codllm/settings/schema.py`. By default, codLLM
expects these files below `data/raw/`:

| Purpose | Filename |
| --- | --- |
| ICD10h masterlist and transfer sheet | `ICD10h_Masterlist_2024.xlsx` |
| Belgium | `SOSA_EXTR_1920-1930 (belgium).xlsx` |
| Amsterdam | `AMC_1854_1926_LM.csv` |
| Copenhagen | `Copenhagen_burials_all_May2025.csv` |
| Ipswich | `Ipswich_deaths_codllm.txt` |
| Madrid | `Madrid 1905_1927.csv` |
| Historic strings | `ICD10H_HISTORICSTRINGSENGLISH_2024.2.xlsx` |

Processed rows are written to `Config.data_processed_dir` as `data.parquet` by default. The processed schema includes
`source_id`, `record_id`, `source_path`, the configured text column, `y_codes`, and the configured label column.

Processed-data caches are validated against source file signatures, source mappings, label harmonization and
standardization settings, input fields, and label settings. Prepared split caches live beside the processed file under:

```text
<processed-stem>.splits/<cache-key>/
```

Those split caches include split-time transformations such as hold-out removal, multi-COD synthesis, balancing, and
masterlist injection.

Inspect cache usage:

```bash
uv run invoke maintenance.data-cache
uv run invoke maintenance.data-cache --lucas
```

Clear processed-data and prepared-split caches with a dry run first:

```bash
uv run invoke maintenance.clear-data-cache
uv run invoke maintenance.clear-data-cache --yes
```

The cache clearer never targets raw datasets. Lock files are kept by default; pass `--locks` only when no training or
cache-build jobs are active.

For broader cleanup, inspect storage first:

```bash
uv run invoke maintenance.status
uv run --no-sync invoke maintenance.status --lucas
```

Then run one of the broad cache policies:

```bash
uv run invoke maintenance.clear-cache --standard
uv run invoke maintenance.clear-cache --standard --days 30 --yes
uv run invoke maintenance.clear-cache --aggressive
uv run invoke maintenance.clear-cache --aggressive --locks --lucas --yes
```

`--standard` targets maintenance-managed generated paths unused for 14+ days by default. `--aggressive` targets all
maintenance-managed generated paths regardless of age. Both modes dry-run unless `--yes` is passed. Broad cleanup may
remove processed-data caches, prepared split caches, local run/checkpoint directories, generated LSF submissions, logs,
Python/tool caches, managed HPC runtime roots such as `$RUN_STORAGE_DIR/cache`, `$RUN_STORAGE_DIR/.venv`, and
`$RUN_STORAGE_DIR/python`, and explicit runtime caches such as Hugging Face, torch, W&B, and uv cache paths. It still
protects raw datasets, source code, tracked TOML specs, arbitrary files under `models/`, and generic profile cache roots
that are not clearly owned by codLLM.

If `hpc.build` fails while creating a `data.splits/*.lock` file with `Disk quota exceeded`, check `maintenance.status`.
When no cache-build or training jobs are active, use `--aggressive --locks --yes` so generated prepared-split roots and
their lock files are removed before rebuilding.

On DTU HPC, source `hpc/env.sh` before running maintenance commands so they inspect the active storage environment.
When checking Lucas's storage from a shell that has not sourced the HPC env, pass `--lucas` to resolve the work folder as
`/work3/s234805` and run storage as `/work3/s234805/codllm`.

On DTU HPC, `maintenance.status` prints shared filesystem capacity as `filesystem_*` values and, when the DTU quota
scripts are available, prints a separate `quota:` line from `getquota_zhome.sh`, `getquota_work1.sh`, or
`getquota_work3.sh`. Capacity errors are usually quota errors, so use the `quota:` line for the enforced user limit.
The storage breakdown includes managed codLLM categories, total `$RUN_STORAGE_DIR`, uncategorized `$RUN_STORAGE_DIR`
usage, the largest direct children of `$RUN_STORAGE_DIR`, large siblings directly under `$STORAGE_FOLDER` that consume
the same quota, and a quota-gap diagnostic when DTU reports more quota usage than is visible below the configured
storage folder.

## Experiment Specs

Experiment specs are TOML files. They can inherit from base specs, set shared environment values in `[env]`, and define
cartesian sweeps in `[sweep]`.

```toml
base = "../base/h100-large.toml"
name = "example-large-run"
command = "train"
force_reprocess = false

[env]
CODLLM_LR = 2.5e-5
CODLLM_NUM_TRAIN_EPOCHS = 8
CODLLM_SAVE_STRATEGY = "no"

[sweep]
CODLLM_LR_SCHEDULER_TYPE = ["cosine", "linear"]
```

TOML booleans become `1` or `0`, arrays in `[env]` become comma-separated strings, and arrays in `[sweep]` expand into
one run per cartesian combination. Use `[[variants]]` for lockstep dimensions that should not cross with each other,
such as pairing a model with its matching H100 runtime base:

```toml
[[variants]]
name = "small"
base = "../profiles/h100-small.toml"

[[variants]]
name = "base"
base = "../profiles/h100-base.toml"
```

Variants cross with `[sweep]`, so a spec with three variants and five hold-out datasets expands to 15 runs instead of a
3x3 accidental model/profile product. Sweep runs also get generated W&B grouping metadata unless explicitly overridden:
`WANDB_SWEEP_ID=codllm-<experiment-name>` and `WANDB_RUN_GROUP=<experiment-name>`.

Useful commands:

```bash
uv run invoke --list
uv run invoke experiments.list
uv run invoke experiments.plan --config runs/sweeps/pretraining.toml --profile h100
```

The checked-in specs are organized by tier under `runs/` (profiles, sweeps, thesis, single). See
[`runs/README.md`](runs/README.md) for the full list and the inheritance design, and the **Reproduce every experiment**
table near the top of this file for the command behind each result.

## HPC Usage

The recommended HPC path is invoke-driven and does not require Docker. Experiment intent stays in `runs/**/*.toml`,
cluster resources stay in `hpc/lsf_profiles.toml`, and generated LSF files are written under `jobs/generated/`.

On an HPC login node, source the storage bootstrap before any `uv` command:

```bash
source hpc/env.sh
bash hpc/storage-check.sh
uv sync --frozen --no-dev
uv run --no-sync invoke hpc.storage
```

`hpc/env.sh` moves uv, Python, Hugging Face, torch, and W&B caches under `$RUN_STORAGE_DIR` instead of personal user
space. After the first successful sync, prefer `uv run --no-sync invoke ...` for planning and submission.

List configured LSF profiles:

```bash
uv run --no-sync invoke hpc.profiles
```

There are two hardware profiles, `h100` and `v100`. Each profile's `wall_time` is the per-slot scheduler ceiling (24 h
for `h100`), not a campaign length — use `--duration` on `hpc.submit` to run across multiple slots (see *Long runs*
below). `h100` requests 17 CPU cores so training specs can use 16 DataLoader workers plus the main process.

Inspect a concrete experiment plan:

```bash
uv run --no-sync invoke experiments.plan \
  --config runs/single/codllm_small.toml \
  --profile h100
```

Prebuild reusable processed-data and prepared-split caches:

```bash
uv run --no-sync invoke hpc.build \
  --config runs/single/codllm_small.toml \
  --profile h100 \
  --user lucas
```

For sweep specs, `hpc.build` builds every expanded run by default. Build one run with a one-based sweep index:

```bash
uv run --no-sync invoke hpc.build \
  --config runs/sweeps/multicod_pretrain.toml \
  --profile h100 \
  --user lucas \
  --sweep-index 2
```

Generate an LSF job without submitting it:

```bash
uv run --no-sync invoke hpc.submit \
  --config runs/sweeps/pretraining.toml \
  --profile h100 \
  --user lucas \
  --dry-run
```

Submit the generated job:

```bash
uv run --no-sync invoke hpc.submit \
  --config runs/sweeps/pretraining.toml \
  --profile h100 \
  --user lucas
```

`--user` currently supports `lucas` and `elias`; `--lucas` and `--elias` are shortcuts. For `hpc.build` and
`hpc.submit`, these aliases set notification email and resolve storage placeholders such as `/work3/$USER` to the
matching DTU account. Sweep specs render as LSF job arrays with one generated env file per array index. The generated
job script sets storage/cache defaults, loads profile modules, runs `uv sync --frozen --no-dev` under a lock unless
`SYNC_ENV=0`, and launches either training or inference from the generated run environment.

HPC defaults:

- raw data: `$PROJECT_DIR/data/raw`
- processed data: `$RUN_STORAGE_DIR/data/processed`
- model outputs: `$RUN_STORAGE_DIR/runs`
- generated scripts/env files/manifests: `jobs/generated/`
- logs: `logs/`

Use the maintenance checks before long HPC runs or before pushing a branch:

```bash
uv run invoke maintenance.status
uv run --no-sync invoke maintenance.status --lucas
uv run --no-sync invoke maintenance.hpc-env --lucas
uv run invoke maintenance.git-hygiene
uv run invoke maintenance.git-snapshot
```

`maintenance.git-snapshot` writes a local `logs/git/` status and recent-commit record. `logs/` is ignored by git, so
cluster stdout/stderr logs and local snapshots do not get pushed accidentally.
`maintenance.status` shows shared filesystem capacity for the workspace, profile/home, and configured HPC storage roots,
then breaks known codLLM usage into code, raw datasets, processed datasets/splits, model outputs, generated jobs/logs,
runtime caches, Python/tool caches, total run storage, uncategorized run storage, and large siblings under the configured
HPC storage folder. With `hpc/env.sh` sourced, it reports the active managed HPC roots; `--lucas` is a shortcut for
Lucas's `/work3/s234805/codllm` storage when the env is not already set. On DTU login nodes, quota
lines come from the DTU quota scripts; the `filesystem_*` totals are shared filesystem capacity and do not represent
the per-user limit that kills jobs.

### Long runs: multi-week campaigns with `--duration`

A single LSF slot is capped by the profile's `wall_time` (24 h on `h100` — the scheduler's hard limit). To train
for longer, add `--duration` to `hpc.submit`: the job trains up to the slot wall time, saves a checkpoint, exits
cleanly (status `0`, not a fake crash), and **resubmits itself** for the next slot until the requested duration is spent
or training converges. Auto-resume continues from the checkpoint each slot; the final slot runs the
test/holdout/uncertainty evaluation exactly once and stops. It stays one logical W&B run throughout, tagged
`codllm/status = needs_resume` between slots and `complete` at the end.

You only pick two things: **the profile** (per-slot resources + wall time) and **the duration** (how long the whole
campaign may run). The submit task derives the per-slot budget from the profile wall time and the resubmission count
from `duration ÷ wall_time`, clamped to the profile's `max_resubmits` safety ceiling (default 30) so a stuck run can
never resubmit forever. Nothing time-related is hardcoded in the spec.

```bash
# Train for up to two weeks, in 24 h slots (~14 slots):
uv run --no-sync invoke hpc.submit \
  --config runs/singles/codllm_base.toml \
  --profile h100 \
  --duration 2w \
  --user lucas
```

`--duration` accepts `w`/`d`/`h`/`m` suffixes (`2w`, `14d`, `48h`; a bare number is hours). Omit `--duration` for a
one-slot run (no resubmission). The command prints the derived campaign (per-slot budget, max resubmissions) before
submitting; the same values are recorded under `submit_overrides` in the submission's `manifest.json`.

The spec only needs auto-resume enabled and, for a long run, an isolated output root so resume always finds *this*
experiment's checkpoint — both already set in [`runs/singles/codllm_base.toml`](runs/singles/codllm_base.toml):

```toml
[env]
CODLLM_OUTPUT_DIR = "runs/codllm_base"        # isolated, resume-stable checkpoint root
CODLLM_AUTO_RESUME = 1                          # continue from the last checkpoint, don't restart
CODLLM_RUNTIME_SAFETY_MARGIN_SECONDS = 600      # save this many seconds before the slot wall limit
```

Lifecycle markers live under `jobs/generated/<submission>/state/run-<index>/` so they survive across slots:

- `.resume_needed` — the slot hit the budget; a checkpoint is saved and the job wants another slot.
- `.training_complete` / `.final_eval_done` — training finished and final evaluation ran once; no further slots.
- `.resubmit_count` / `.resubmit_exhausted` — the resubmission counter and the marker written when the guard trips.

`CODLLM_AUTO_RESUME` is mandatory: without it a resubmitted slot would restart from scratch, so the script refuses to
resubmit and warns instead. A real crash trips the job's error trap and exits non-zero, so crashes never resubmit. The
per-slot budget is measured from job start (the script exports `CODLLM_JOB_START_EPOCH`), so setup time counts against
the wall limit and the checkpoint is always saved before the scheduler kills the slot. Raise a profile's `max_resubmits`
in [`hpc/lsf_profiles.toml`](hpc/lsf_profiles.toml) if a campaign needs more slots than the ceiling allows.

## Training Options

All runtime behavior is owned by `Config` in `src/codllm/settings/schema.py` and environment overrides in
`src/codllm/settings/env.py`. Prefer setting options in TOML specs rather than editing code.

### Model Size and Task

- `CODLLM_HF_MODEL` selects a Hugging Face model id or local checkpoint path.
- `runs/profiles/h100-small.toml` uses `google/flan-t5-small`.
- `runs/profiles/h100-large.toml` uses `google/flan-t5-large`.
- `CODLLM_MODEL_TASK=seq2seq` is the default and supports single-label and multi-COD targets.
- `CODLLM_MODEL_TASK=sequence_classification` uses `AutoModelForSequenceClassification` and currently requires
  `CODLLM_MAX_LABEL_COUNT=1`.
- `CODLLM_TORCH_DTYPE` accepts `auto`, `float16`, `bfloat16`, or `float32`.
- `CODLLM_LOAD_IN_8BIT=1` is supported for model loading on CUDA, but the current full fine-tuning path rejects
  quantized trainable models.

### Input Fields and Labels

- `CODLLM_TRAINING_INPUT` controls the processed input fields: `cod`, `age`, and `sex`.
- `CODLLM_INPUT_PREFIX_COD`, `CODLLM_INPUT_PREFIX_AGE`, and `CODLLM_INPUT_PREFIX_SEX` control text prefixes.
- `CODLLM_TEXT_FIELD_SEPARATOR` joins input fields.
- `CODLLM_LABEL_SEPARATOR` joins multi-label targets.
- `CODLLM_MAX_LABEL_COUNT` controls how many ICD10h labels a row may carry.
- `CODLLM_MAX_TARGET_LENGTH`, `CODLLM_LABEL_CODE_LENGTH`, and `CODLLM_MAX_TARGET_LENGTH_BUFFER` control target
  generation length; the effective target length is raised automatically when multi-label targets need more room.

### Label Harmonization

Label harmonization is enabled by default. It loads the configured ICD10h masterlist workbook, maps 2020 labels through
the transfer sheet, normalizes labels to the `A00.000` shape, and drops rows whose labels are absent from the
masterlist.

Main controls:

- `CODLLM_LABEL_HARMONIZATION_ENABLED`
- `CODLLM_LABEL_HARMONIZATION_MASTERLIST_PATH`
- `CODLLM_LABEL_HARMONIZATION_MASTERLIST_SHEET_NAME`
- `CODLLM_LABEL_HARMONIZATION_TRANSFER_SHEET_NAME`

### Label Standardization

Label standardization is enabled by default as a tracked no-op overlay under `data/curation/`. Keep raw datasets
unchanged. Add reviewed dataset-level rules to `data/curation/label_standardization.toml` and exact row-level exceptions
to `data/curation/label_standardization_overrides.csv`. The overlay runs after masterlist harmonization and is included
in processed-data cache metadata, so curation edits trigger a rebuild.

Main controls:

- `CODLLM_LABEL_STANDARDIZATION_ENABLED`
- `CODLLM_LABEL_STANDARDIZATION_RULES_PATH`
- `CODLLM_LABEL_STANDARDIZATION_OVERRIDES_PATH`

### Masterlist Pretraining

Set `CODLLM_PRETRAIN_ENABLED=1` to run a two-stage flow: pretrain on ICD10h masterlist strings, then fine-tune on the
prepared historical splits. Pretraining has separate learning-rate, warmup, scheduler, epoch, evaluation, upsampling,
and synthetic multi-COD controls, so it does not reuse fine-tuning knobs by accident.

Common controls:

- `CODLLM_PRETRAIN_MASTERLIST_PATH`
- `CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS`
- `CODLLM_PRETRAIN_LEARNING_RATE`
- `CODLLM_PRETRAIN_WARMUP_RATIO`
- `CODLLM_PRETRAIN_LR_SCHEDULER_TYPE`
- `CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS`
- `CODLLM_PRETRAIN_UPSAMPLE_ENABLED`
- `CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL`
- `CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS`
- `CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE`
- `CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_RATIO`
- `CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_TEXT_SEPARATOR`

### Multi-COD

Multi-COD training is enabled by setting `CODLLM_MAX_LABEL_COUNT` above `1`. For seq2seq runs, target labels are emitted
as separator-joined ICD10h strings.

Controls:

- `CODLLM_MULTICOD_SHUFFLE_LABELS` shuffles multi-label target order deterministically from the data seed.
- `CODLLM_MULTICOD_SYNTHETIC_RATIO` creates training-only synthetic multi-COD rows from eligible single-COD rows.
- `CODLLM_MULTICOD_SYNTHETIC_SOURCE_SCOPE` is `within_source` by default; `any_source` is opt-in.
- `CODLLM_MULTICOD_SYNTHETIC_TEXT_SEPARATOR` joins merged cause strings.

Synthetic multi-COD rows are added only after train/validation/test splitting, and only to the training split.

### Balancing and Perturbation

Balancing is a training-split transformation. It never changes validation, test, or final hold-out rows.

The pipeline has two independent stages. Floor upsampling uses `CODLLM_BALANCE_*`; whole-training-set base
perturbation uses `CODLLM_BASE_PERTURBATION_*`. Either or both may be active in a single run; setting
`CODLLM_BALANCE_STRATEGY=none` and `CODLLM_BASE_PERTURBATION_RATE=0` disables both.

#### Stage 1 — Floor upsampling (`CODLLM_BALANCE_STRATEGY=floor`)

Every class with fewer than `CODLLM_BALANCE_FLOOR` samples is upsampled by drawing additional rows from the same class
until the floor is reached. The new rows are perturbed proportionally — the more aggressive the upsample, the more
likely each synthetic copy is to be perturbed (so a class that was 10× upsampled receives noisier copies than one that
was only doubled). `CODLLM_BALANCE_FLOOR_DECAY` softens the floor for the rarest classes via a log curve so a class
with 1 original sample still ends up smaller than a class with 5 originals after upsampling — the natural ranking is
preserved. Set `_DECAY=0` for a strict floor.

Worked example with `FLOOR=50, DECAY=0`:

| Original count | Target after upsampling |
|---|---|
| 1 | 50 |
| 5 | 50 |
| 49 | 50 |
| 50 | 50 (unchanged) |
| 200 | 200 (unchanged) |

#### Stage 2 — Base-rate perturbation (`CODLLM_BASE_PERTURBATION_RATE > 0`)

After upsampling, a configurable fraction of the *entire training set* (synthetic + original) is perturbed in place.
This is regularization, not balance correction — `RATE=0.0` is the safe default, `RATE=1.0` perturbs every row. The
2026-04 sweep on flan-t5-small showed `RATE=1.0` hurts macro_f1 by ~5.7pp because the model never sees clean training
text but is evaluated on clean text. Reach for low rates (0.05 – 0.3) when you want regularization without distribution
shift.

#### Perturbation mechanics

Perturbations are applied only to the configured `cod` segment of the training text — never to age, sex, or other
metadata fields, so structural metadata stays intact across synthetic copies.

- `CODLLM_BALANCE_PERTURBATIONS` — comma-separated names from {`swap_adjacent_chars`, `delete_random_char`,
  `insert_random_whitespace`, `accent_random_vowel`, `qwerty_misspell`}. Applies to floor-upsampled copies.
- `CODLLM_BALANCE_PERTURBATION_MEAN`, `_VARIANCE`, and `_LOFT` — control edits for floor-upsampled copies.
- `CODLLM_BASE_PERTURBATIONS`, `CODLLM_BASE_PERTURBATION_MEAN`, `_VARIANCE`, and `_LOFT` — equivalent controls for the
  whole-training-set regularization pass. Counts scale with COD text length, so longer cause strings receive more edits
  on average. With variance enabled, `_LOFT` caps the sampled count at
  `mean * cod_length + loft * sqrt(variance * cod_length)`, preventing rare long-tail samples from making a string far
  noisier than intended. The default loft is `3.0`.

### Hold-Out Evaluation

Set `CODLLM_HOLD_OUT_DATASET` to a processed `source_id` for leave-one-source-out evaluation. Matching rows are removed
before dataset sampling and train/validation/test splitting. After training, the full held-out source is evaluated with
canonical `holdout/full/*` W&B metrics plus legacy `holdout/*` aliases, while regular validation remains under `val/*`.

Set `CODLLM_TRAIN_EXCLUDED_SOURCE_IDS` to a comma-separated list of processed `source_id` values that should be removed
from the train/validation/test pool without becoming the held-out evaluation set. Hold-out sweeps should normally set
`CODLLM_TRAIN_EXCLUDED_SOURCE_IDS=historic_strings_en_2024` so the reference historic strings source is not used as
ordinary fine-tuning data.

During-training hold-out evaluation is optional:

- `CODLLM_HOLD_OUT_EVALUATE_PER=epoch|steps`
- `CODLLM_HOLD_OUT_EVALUATE_RATIO=0.05`

The during-training callback evaluates a deterministic sample of the held-out source. The final post-training hold-out
evaluation always uses the full held-out source. Sampled during-training hold-out metrics are logged under
`holdout/sample/*` plus legacy `holdout/val/*` aliases.

Default source ids:

- `belgium_1920_1930`
- `amsterdam_1854_1926`
- `copenhagen_may2025`
- `ipswich_1871_1911`
- `madrid_1905_1927`
- `historic_strings_en_2024`

### Masterlist Injection

`CODLLM_MASTERLIST_INJECT_ENABLED=1` injects masterlist-derived rows into the training split after balancing. This is
separate from pretraining and has its own target-per-label and perturbation controls:

- `CODLLM_MASTERLIST_INJECT_TARGET_PER_LABEL`
- `CODLLM_MASTERLIST_INJECT_PERTURBATIONS`
- `CODLLM_MASTERLIST_INJECT_PERTURBATIONS_PER_SAMPLE`

### Training Lifecycle and HPC Resilience

These controls govern checkpoint cadence, early stopping, wall-time budgeting, auto-resume across LSF slots, and the
optional end-of-training uncertainty pass. They are intended for long training runs that need to survive job restarts.

#### Checkpoint and evaluation cadence

- `CODLLM_EVAL_STRATEGY=epoch|steps` controls when validation runs.
- `CODLLM_SAVE_STRATEGY=epoch|steps` controls when checkpoints are saved.
- `CODLLM_EVAL_STEPS` and `CODLLM_SAVE_STEPS` set the step interval when using the `steps` strategy.
- `CODLLM_SAVE_TOTAL_LIMIT` caps the number of retained checkpoints. With `LOAD_BEST_MODEL_AT_END=1` the trainer keeps
  the best one plus the most recent up to this limit.
- `CODLLM_LOGGING_STEPS` sets how often training-side `train/loss` lines are emitted.

#### Best-model selection and early stopping

- `CODLLM_LOAD_BEST_MODEL_AT_END=1` reloads the best checkpoint before final test evaluation.
- `CODLLM_SAVE_STRATEGY_BEST_METRIC` chooses which validation metric ranks checkpoints. Common values: `macro_f1`,
  `sample_f1`, `accuracy`.
- `CODLLM_EARLY_STOPPING_PATIENCE` is the number of consecutive evaluations without improvement that triggers a stop.
  `0` disables early stopping.
- `CODLLM_EARLY_STOPPING_THRESHOLD` is the minimum improvement that resets the patience counter. `0.0` treats any
  strict improvement as fresh progress (a "best yet" criterion).
- `CODLLM_PRETRAIN_EARLY_STOPPING_PATIENCE` and `CODLLM_PRETRAIN_LOAD_BEST_MODEL_AT_END` override those two controls
  for the masterlist-pretraining stage. When unset, they inherit the fine-tuning values. Set them to `0` and `false`
  for an exact pretraining-dose experiment.
- `CODLLM_FINAL_TEST_EVAL_ENABLED=false` keeps the configured test split intact but suppresses final test scoring and
  test-based uncertainty evaluation. Use this for response curves and hyperparameter search to avoid test leakage.

#### Wall-time budgeting

LSF slots have a hard wall (typically 24h on `gpuh100`). The trainer can shut down gracefully before the wall via a
runtime callback that periodically checks elapsed time.

- `CODLLM_MAX_RUNTIME_SECONDS` is the upper bound on this slot's training time. Once reached the callback signals the
  trainer to stop after the current step, runs `load_best_model_at_end` + final test evaluation, and writes a
  `.resume_needed` marker so the next slot resumes cleanly. `0` disables the budget.
- `CODLLM_RUNTIME_SAFETY_MARGIN_SECONDS` is the headroom kept BEFORE `MAX_RUNTIME_SECONDS` to ensure the final
  evaluation and checkpoint save complete before SIGKILL. Default 300s. Raise this for large models where the end-of-
  training evaluation pass is slow (XL with `max_label_count=3` and a 76k test split can need 30-60 min).

#### Auto-resume across slots

- `CODLLM_AUTO_RESUME=1` makes the trainer pick up from the latest checkpoint in the output directory on startup. Used
  with the LSF resubmit flow so each new slot continues the same logical training run.
- `CODLLM_PER_SIZE_OUTPUT_DIR=1` puts each model-size run under its own subdirectory so size-sweep slots do not
  collide. Combined with auto-resume, this lets multiple sizes resubmit independently.
- The W&B sidecar `wandb_run_id.txt` is written in the resume-stable `CODLLM_RUN_STATE_DIR` and reread on resume so
  fresh scheduler slots write into the same W&B run. Checkpoint-local sidecars from older runs remain readable.
- Intermediate wall-time stops do not upload a misleading final-model artifact. This leaves the safety margin
  available for checkpointing and a clean W&B telemetry flush before resubmission.

#### End-of-training uncertainty pass

After the final test evaluation, the trainer can run a per-record uncertainty pass on the test split (temperature
fit on validation logits, then per-record signals: sum/mean/min logprob, mean and first-token entropy, risk-coverage
curves). The artifacts (`temperature.json`, `predictions.jsonl`, `test_rows.parquet`, `rc_curves.json`) are logged
to W&B as an `end_of_training_eval` artifact.

- `CODLLM_UNCERTAINTY_EVAL=true|false` toggles the pass. Default is `true`.
- The pass is wrapped in a Python `try/except`, but a native-code SIGFPE that has been observed on large models will
  still terminate the process (the catch never fires). If you do not need the uncertainty artifacts on a given run,
  set `CODLLM_UNCERTAINTY_EVAL=false`. The test metrics from the final evaluation are written BEFORE this pass, so
  disabling it does not affect `test/macro_f1`, the saved best checkpoint, or any other training output.
- `CODLLM_UNCERTAINTY_EVAL_ENABLED` is the legacy name for the same flag and still works, but logs a deprecation
  warning. Migrate any existing TOMLs to the new name.

### Weights and Biases

Logging is automatic when W&B credentials are present. All knobs are overridable from env or TOML:

- `CODLLM_WANDB_ENABLED=0` disables W&B entirely for this run.
- `CODLLM_WANDB_MODE=auto|online|offline|disabled` controls the W&B client mode. `auto` uses online mode when
  credentials are present; `offline` writes data to disk only and requires a later `wandb sync` to upload.
- `CODLLM_WANDB_PROJECT` and `CODLLM_WANDB_ENTITY` select the destination.
- `CODLLM_WANDB_RUN_NAME` overrides the auto-generated run name.
- `CODLLM_WANDB_RUN_CONFIG_MODE=minimal|standard|full` controls how much of the resolved `Config` is shown on the
  W&B run page (with the full version always available as a logged artifact).
- `CODLLM_WANDB_METRIC_MODE=core|standard|all` filters which metric scopes are logged: `core` keeps the headline
  numbers and skips per-chapter / per-block / per-source breakdowns, `all` logs everything.
- `CODLLM_WANDB_LOG_MODEL=false|end|checkpoint` controls whether model checkpoints are uploaded as artifacts.
- High-level codLLM progress messages are sent directly to the W&B Logs tab, including after a scheduler-slot resume.
  The run summary records the active LSF job id/index and `codllm/status`.

## Outputs and Metrics

Each training run writes checkpoints under a run-scoped output directory. Locally this becomes `runs/run-0001`,
`runs/run-0002`, and so on. On LSF, run ids use `LSB_JOBID` and `LSB_JOBINDEX`.

When W&B credentials are available, training logs:

- compact run-page config selected by `CODLLM_WANDB_RUN_CONFIG_MODE=minimal|standard|full`
- full resolved config, runtime metadata, source metadata, and training args as run metadata artifacts
- source file metadata and processed-cache fingerprints
- split sizes, source distributions, and label summaries
- compact data visualizations under `data/*`
- validation, test, hold-out, and pretraining metrics under scoped namespaces, filtered by
  `CODLLM_WANDB_METRIC_MODE=core|standard|all`
- hold-out leakage diagnostics such as `dataset.holdout_leakage.input_seen_rate` on the W&B run page and in metadata
  artifacts
- full evaluation metrics and row-level predictions, including row `source_id`, as artifacts, plus compact error tables
  under scopes such as `val/errors/*`, `test/errors/*`, `holdout/full/errors/*`, and sampled
  `holdout/sample/errors/*`

Multi-COD runs emit exact-match, micro, sample, Jaccard, Hamming, and label-count diagnostics. Single-label runs emit
accuracy and macro precision/recall/F1.

## Inference

Run inference through the package entrypoint:

```bash
uv run python -m codllm.inference data/inference/input.csv \
  --hf-model runs/run-0001/checkpoint-500 \
  --output-path runs/inference/predictions.csv
```

For pretraining-enabled runs, use a checkpoint from the final `finetune/` stage.

Input formats:

- `.txt`: one example per line
- `.csv`, `.tsv`, `.jsonl`, `.parquet`: must contain the configured text column, `text` by default

Output formats are selected by extension: `.csv`, `.tsv`, `.jsonl`, or `.parquet`. Without `--output-path`, predictions
are written as compact JSONL to stdout.

Useful inference flags:

- `--hf-model` overrides `CODLLM_HF_MODEL` with a model id or local checkpoint path.
- `--model-task seq2seq|sequence_classification` overrides the configured task.
- `--validate-registry` rejects predicted codes absent from the configured ICD10h masterlist.

The matching environment variable for registry validation is `CODLLM_INFERENCE_VALIDATE_REGISTRY=1`.

## Docker

Docker is mainly for local containerized training, not the recommended HPC path.

```bash
docker build -f dockerfiles/train.dockerfile -t codllm-train:latest .
```

```bash
docker run --rm \
  -v codllm-runs:/app/runs \
  -v codllm-processed:/app/data/processed \
  -v "$PWD/data/raw:/app/data/raw:ro" \
  -e HUGGINGFACE_HUB_TOKEN \
  -e WANDB_API_KEY \
  codllm-train:latest
```

## Development Commands

```bash
uv run pytest tests/
uv run ruff format .
uv run ruff check . --fix
uv run invoke --list
```

The GitHub Actions test workflow runs `uv sync --frozen --all-groups` and `uv run pytest tests/ -q` on Linux.

## License

This repository is licensed under the MIT License.

The historical datasets used for training and evaluation are not public and require separate agreements with the data
providers.
