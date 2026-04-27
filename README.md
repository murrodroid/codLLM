# Coding Historical Causes of Death with Large Language Models

This project investigates whether modern Large Language Models (LLMs) can automatically
translate historical free-text descriptions of causes of death into ICD10h codes.

Historical demographers currently perform this coding manually. The process is time-consuming,
requires specialised knowledge, and does not scale to rapidly growing digitised archives.
This project aims to reduce coding time from months to minutes using machine learning.

## Project Goals

- Develop a training pipeline for fine-tuning an LLM on historical cause-of-death data.
- Evaluate model accuracy and robustness across cities, languages, and time periods.
- Build a simple prototype tool for applying the model to new datasets, including low-resource settings.

## Data

This project builds on existing European historical datasets, including manually coded
cause-of-death records with ICD10h classifications.

> **Important:** The datasets used in this project are not included in this repository and
> are subject to separate data-sharing agreements with the respective institutions.

## Method Overview

The pipeline consists of:

1. Data preprocessing and normalization
2. Fine-tuning of a transformer-based language model
3. Cross-city and cross-period evaluation
4. Robustness analysis
5. Prototype inference interface

## Repository Structure

```
codLLM/
├── main.py                         # Main entry point
├── pyproject.toml                  # Project metadata and dependencies
├── data/                           # Data directory (no raw data included)
├── dockerfiles/
│   └── train.dockerfile
├── models/
│   └── placeholder.pth
├── src/
│   └── codllm/
│       ├── config.py               # Project configuration
│       ├── data_augmentation.py    # Data augmentation utilities
│       ├── data_handler.py         # Dataset loading and mapping
│       ├── model_registry.py       # Model loader registry
│       ├── preprocess.py           # Tokenization preprocessing
│       └── train.py                # Training scripts
└── tests/
    └── test_training.py            # Training tests
```

## Installation

```bash
git clone https://github.com/murrodroid/codllm
cd codllm
pip install uv
uv sync
```

## Inference

Run inference directly through the package entrypoint:

```bash
uv run python -m codllm.inference data/inference/input.csv --output-path runs/inference/predictions.csv
```

Input formats:

- `.txt`: one inference example per line
- `.csv`, `.tsv`, `.jsonl`, `.parquet`: must contain the configured text column (`text` by default)

When `--output-path` is omitted, the CLI writes compact JSONL predictions to stdout.
Set `CODLLM_INFERENCE_VALIDATE_REGISTRY=1` to reject predicted codes that are absent from the configured ICD10h
masterlist.

## Runtime Environment Variables

Environment variables are supported, but only if they are explicitly read by the scripts.

- `jobs/train.sh` reads a fixed set of variables and runs on HPC.
- The Python training code reads `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) and `WANDB_API_KEY`.
- Training and reproducibility controls are read from `CODLLM_*` env vars (listed below).
- Arbitrary env vars are ignored unless the code references them.

## Reproducibility Defaults

The training runtime uses config-driven reproducibility defaults:

- global seed (`seed`)
- data split/sampler seed (`data_seed`, defaults to `seed`)
- deterministic torch algorithms
- deterministic CuDNN mode
- fixed dataloader worker count (`4` by default)
- dataloader prefetch queue depth (`2` by default)
- dynamic target-length floor for labels

You can override these at runtime without editing code:

```bash
export CODLLM_SEED=42
export CODLLM_DATA_SEED=42
export CODLLM_DATALOADER_NUM_WORKERS=4
export CODLLM_DATALOADER_PIN_MEMORY=true
export CODLLM_DATALOADER_PERSISTENT_WORKERS=false
export CODLLM_DATALOADER_PREFETCH_FACTOR=2
export CODLLM_HF_MODEL=google/flan-t5-small
export CODLLM_MAX_SOURCE_LENGTH=256
export CODLLM_MAX_TARGET_LENGTH=32
export CODLLM_DETERMINISTIC_ALGORITHMS=true
export CODLLM_CUDNN_DETERMINISTIC=true
export CODLLM_CUDNN_BENCHMARK=false
export CODLLM_LOAD_IN_8BIT=0
export CODLLM_VERBOSE=true
export CODLLM_TORCH_DTYPE=auto
export CODLLM_WANDB_LOG_MODEL=end
export CODLLM_WARMUP_STEPS=1000
export CODLLM_NUM_TRAIN_EPOCHS=4
export CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE=8
export CODLLM_PER_DEVICE_EVAL_BATCH_SIZE=8
export CODLLM_GRADIENT_ACCUMULATION_STEPS=2
export CODLLM_LOGGING_STEPS=25
export CODLLM_EVAL_STEPS=200
export CODLLM_SAVE_STEPS=5000
export CODLLM_EVAL_STRATEGY=epoch
export CODLLM_SAVE_STRATEGY=epoch
export CODLLM_SAVE_STRATEGY_BEST_METRIC=macro_f1
export CODLLM_LR=3e-5
export CODLLM_WEIGHT_DECAY=0.0
export CODLLM_MAX_GRAD_NORM=0.5
export CODLLM_TRAINING_INPUT="cod,age,sex"
export CODLLM_MAX_LABEL_COUNT=2
export CODLLM_LABEL_CODE_LENGTH=7
export CODLLM_LABEL_SEPARATOR=" | "
export CODLLM_MAX_TARGET_LENGTH_BUFFER=4
export CODLLM_PRETRAIN_ENABLED=1
export CODLLM_PRETRAIN_MASTERLIST_PATH=data/raw/ICD10h_Masterlist_2024.xlsx
export CODLLM_PRETRAIN_MASTERLIST_SHEET_NAME=Masterlist
export CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS=1
export CODLLM_PRETRAIN_UPSAMPLE_ENABLED=1
export CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL=10
export CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS=swap_adjacent_chars,delete_random_char,accent_random_vowel,qwerty_misspell
export CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE=1
```

## Docker (Local)

Build:

```bash
docker build -f dockerfiles/train.dockerfile -t codllm-train:latest .
```

Run:

```bash
docker volume create codllm-runs
docker volume create codllm-processed

docker run --rm \
  -v codllm-runs:/app/runs \
  -v codllm-processed:/app/data/processed \
  -e HUGGINGFACE_HUB_TOKEN \
  -e WANDB_API_KEY \
  -e CODLLM_SEED=42 \
  -e CODLLM_DATA_SEED=42 \
  codllm-train:latest
```

`WANDB_API_KEY` is optional. Without it, training runs with W&B disabled.
Model outputs and processed data persist in the named Docker volumes.
Each training invocation writes checkpoints under a run-scoped folder:
`<CODLLM_OUTPUT_DIR>/run-<id>/checkpoint-*`. On HPC, `<id>` uses `LSB_JOBID`
(and `LSB_JOBINDEX` when present). Locally, `<id>` is an auto-incremented number.

## Feature Implementations

### Data Augmentation and Upsampling

Balancing is controlled through `Config` (or matching `CODLLM_*` env vars):

- `balance_strategy`: `"none"` disables upsampling, `"upsample"` enables class-count upsampling.
- `balance_target_quantile`: quantile used to compute the target class count for upsampling.
- `balance_upsample_labels`: optional allow-list of labels that may be upsampled (empty = all eligible minority labels).
- `balance_upsample_inverse_power`: inverse-frequency scaling exponent in `(0, 1]`; higher values boost smaller minority classes more.
- `balance_upsample_budget_ratio`: synthetic-row budget as a ratio of training-set size (`0..1`), used to scale upsampling dynamically.
- `balance_perturbations`: augmentation functions applied to the `cod:` text segment.
- `balance_perturbations_per_sample`: number of perturbations chained per affected sample.
- `balance_base_perturbation_rate`: chance/rate (`0..1`) to perturb all training rows after upsampling.

Typical setups:

- Minority upsampling + global perturbation: set `balance_strategy="upsample"` and tune `balance_target_quantile`, `balance_upsample_inverse_power`, `balance_upsample_budget_ratio`, and `balance_base_perturbation_rate`.
- Global perturbation without upsampling: set `balance_strategy="none"` and `balance_base_perturbation_rate > 0`.

Runtime order: minority labels are upsampled first using random row draws from each minority class, then perturbations are applied across the resulting training rows.

### Masterlist Pretraining

You can run a two-stage pipeline where the model first pretrains on the ICD10h masterlist and then continues normal training.

Pretraining-specific knobs:

- `CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS` controls pretraining epochs.
- `CODLLM_PRETRAIN_LEARNING_RATE` optionally overrides pretraining LR (falls back to `CODLLM_LR`).
- `CODLLM_PRETRAIN_LR_SCHEDULER_TYPE` controls the pretraining scheduler (default: `linear`).
- `CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS` runs pretraining validation every N epochs (final epoch is always evaluated).
- `CODLLM_PRETRAIN_UPSAMPLE_ENABLED` enables label-wise pretraining upsampling (default: enabled).
- `CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL` sets the pretraining target rows per label (default: `10`).
- `CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS` sets perturbation functions for synthetic pretraining rows.
- `CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE` sets perturbation chain depth per synthetic row.
- Pretraining warmup is fixed to `0` steps.
- Fine-tuning starts a new Trainer stage, so LR scheduler steps reset from the configured fine-tuning LR.
- For sequence classification, set `CODLLM_MODEL_TASK=sequence_classification`; class ids are built from the masterlist `ICD10h` values.
- Run metadata includes `pretraining.upsampling` diagnostics such as `rows_added`, `perturbation_rate`, and label-count summaries.

## HPC Usage (LSF, No Docker)

Use this path when your cluster does not allow Docker.

The recommended HPC workflow is now invoke-driven:

- experiment intent lives in TOML specs under `runs/`
- cluster resources live in named profiles in `hpc/lsf_profiles.toml`
- generated LSF scripts and per-run env files are written to `jobs/generated/`
- generated artifacts are ignored by git and can be inspected before submission

List available tasks:

```bash
uv run invoke --list
```

On HPC login nodes, source the storage bootstrap before running any `uv` command. This ensures uv packages, the project
virtualenv, uv-managed Python installs, Hugging Face caches, torch caches, and W&B caches are placed under the storage
unit instead of personal user space:

```bash
source hpc/env.sh
bash hpc/storage-check.sh
uv sync --frozen
uv run --no-sync invoke hpc.storage
```

After `uv sync --frozen` has succeeded once, use `uv run --no-sync invoke ...` for planning and submission commands so
uv does not unexpectedly resync while you are only inspecting specs.

List experiment specs and profiles:

```bash
uv run --no-sync invoke experiments.list
uv run --no-sync invoke hpc.profiles
```

Inspect the concrete runs created by a spec:

```bash
uv run --no-sync invoke experiments.plan --config runs/sweeps/pretraining.toml --profile h100-10h
```

Generate an LSF submission without submitting it:

```bash
uv run --no-sync invoke hpc.submit \
  --config runs/sweeps/pretraining.toml \
  --profile h100-10h \
  --user lucas \
  --dry-run
```

Submit the generated job:

```bash
uv run --no-sync invoke hpc.submit \
  --config runs/sweeps/pretraining.toml \
  --profile h100-10h \
  --user lucas
```

For sweep specs, the generated script uses an LSF job array and one generated env file per array index. The Python
training code still receives ordinary `CODLLM_*` environment variables through `config_from_env`, so the model runtime
does not need to know whether a run came from a local shell, an invoke task, or LSF.

### Experiment Specs

Experiment specs are TOML files. They can inherit from one or more base specs, define scalar env overrides, and define
cartesian sweeps:

```toml
base = "../base/h100-large.toml"
name = "pretraining-epochs"
command = "train"
force_reprocess = false

[env]
CODLLM_MODEL_TASK = "seq2seq"
CODLLM_PRETRAIN_ENABLED = true

[sweep]
CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS = [1, 10, 25]
```

Use `[env]` for values that should be present in every run. Use `[sweep]` for values that should expand to multiple
runs. TOML booleans are converted to `1` or `0`, arrays in `[env]` are converted to comma-separated strings, and arrays
in `[sweep]` expand into individual runs.

### LSF Profiles

LSF profiles keep scheduler details out of experiment specs. A profile defines queue, wall time, CPU/GPU resources,
memory, module loads, storage defaults, and log locations. Edit `hpc/lsf_profiles.toml` when moving between queues or
clusters instead of changing experiment specs.

### Legacy shell wrappers

The older handwritten shell wrappers remain available for compatibility, but new experiments should prefer the
invoke/TOML workflow.

Script: `jobs/train.sh`
H100 script: `jobs/train_h100.sh`

### 1) Edit scheduler directives in `jobs/train.sh`

Update the `#BSUB` lines for your cluster before first run:

- queue (`-q`)
- wall time (`-W`)
- CPU/GPU request (`-n`, `-gpu`)
- memory (`-R "rusage[mem=...]"`)
- email (`-u`)
- output log path (`-oo`)

### 2) Prepare environment and submit

From repo root:

```bash
mkdir -p logs

export STORAGE_FOLDER="/work3/$USER"
export HUGGINGFACE_HUB_TOKEN="YOUR_HF_KEY"   # optional, but recommended
export WANDB_API_KEY="YOUR_WANDB_KEY"        # optional
export CODLLM_SEED=42
export CODLLM_DATA_SEED=42

bsub < jobs/train.sh
```

Note: save variables permanently in shell with this logic:

```bash
echo 'export MY_VAR="my_value"' >> ~/.zshrc
source ~/.zshrc
```

Some clusters do not forward temporary `VAR=value bsub ...` values unless `-env` is used.
Use `-env "all"` to make forwarding explicit.

You can also submit using a job config file:

```bash
bsub -env "all,JOB_CONFIG_FILE=jobs/configs/example.env,REQUIRE_JOB_CONFIG_FILE=1" < jobs/train.sh
```

`JOB_CONFIG_FILE` accepts simple shell `KEY=value` lines (comments with `#` are allowed).
See `jobs/configs/example.env`.
You can define model selection and all training knobs here, for example
`CODLLM_HF_MODEL`, `CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE`, and `CODLLM_GRADIENT_ACCUMULATION_STEPS`.
You can also run a single-variable sweep by using list syntax:

```bash
CODLLM_NUM_TRAIN_EPOCHS=[2,4,8]
```

Only one list-valued variable is supported per config file.

For `jobs/train_h100.sh`, submit in the same style as `jobs/train.sh`:

```bash
bsub -env "all,JOB_CONFIG_FILE=jobs/configs/t5-large_h100.env,REQUIRE_JOB_CONFIG_FILE=1" < jobs/train_h100.sh
```

For job arrays or many concurrent runs, shared processed-data writes are now lock-protected.
You should normally keep `FORCE_REPROCESS=0` so workers reuse the cache when metadata matches.

### 3) Monitor

```bash
bjobs
tail -f logs/<job_id>.out
```

### Native HPC env vars supported by `jobs/train.sh`

- `STORAGE_FOLDER` (default: `/work3/s234805`)
- `RUN_STORAGE_DIR` (default: `$STORAGE_FOLDER/codllm`)
- `TRAIN_DATA_RAW_DIR` (default: `$PROJECT_DIR/data/raw`)
- `TRAIN_DATA_PROCESSED_DIR` (default: `$RUN_STORAGE_DIR/data/processed`)
- `TRAIN_OUTPUT_DIR` (default: `$RUN_STORAGE_DIR/runs`)
- `JOB_CONFIG_FILE` (optional path to a shell-style job config file)
- `REQUIRE_JOB_CONFIG_FILE` (`1` to fail early if `JOB_CONFIG_FILE` is missing, default `0`)
- `CODLLM_DATA_RAW_DIR`, `CODLLM_DATA_PROCESSED_DIR`, `CODLLM_OUTPUT_DIR` (optional overrides)
- `SYNC_ENV` (`1` to run `uv sync`, default `1`)
- `UV_SYNC_LOCK_FILE` (lock file used to serialize `uv sync`, default: `$RUN_STORAGE_DIR/.uv-sync.lock`)
- `FORCE_REPROCESS` (`1` adds `--force-reprocess`, default `0`)
- `TRAIN_EXTRA_ARGS` (optional args appended to `python -m codllm.training`)
- `HUGGINGFACE_HUB_TOKEN`, `WANDB_API_KEY`, `WANDB_MODE`
- `CODLLM_*` training/reproducibility settings from the section above
- `CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS` (processed-cache lock wait timeout, default: `900`)
- `CODLLM_RUN_DIR_LOCK_TIMEOUT_SECONDS` (run-dir lock wait timeout, default: `120`)
- `CODLLM_LOAD_IN_8BIT` (`0` by default in `jobs/train.sh`)
- `CODLLM_VERBOSE` (`1`/`0`, default: `1`; prints resolved setup before training)
- `CODLLM_TORCH_DTYPE` (`auto`, `float16`, `bfloat16`, `float32`)
- `CODLLM_HF_MODEL` (default: `google/flan-t5-small`)
- `CODLLM_MAX_SOURCE_LENGTH` (default: `256`)
- `CODLLM_MAX_TARGET_LENGTH` (default: `32`)
- `CODLLM_WANDB_LOG_MODEL` (`false`, `end`, `checkpoint`; default: `end`)
- `CODLLM_WARMUP_STEPS` (default: `1000`)
- `CODLLM_NUM_TRAIN_EPOCHS` (default: `4`)
- `CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE` (default: `8`)
- `CODLLM_PER_DEVICE_EVAL_BATCH_SIZE` (default: `8`)
- `CODLLM_GRADIENT_ACCUMULATION_STEPS` (default: `2`)
- `CODLLM_DATALOADER_NUM_WORKERS` (default: `4`)
- `CODLLM_DATALOADER_PIN_MEMORY` (`1`/`0`; default: `1`)
- `CODLLM_DATALOADER_PERSISTENT_WORKERS` (`1`/`0`; default: `0`)
- `CODLLM_DATALOADER_PREFETCH_FACTOR` (default: `2`; only applied when workers > `0`)
- `CODLLM_LOGGING_STEPS` (default: `25`)
- `CODLLM_EVAL_STEPS` (default: `200`)
- `CODLLM_SAVE_STEPS` (default: `5000`)
- `CODLLM_EVAL_STRATEGY` (`no`, `steps`, `epoch`; default: `epoch`)
- `CODLLM_SAVE_STRATEGY` (`no`, `steps`, `epoch`, `best`; default: `epoch`)
- `CODLLM_SAVE_STRATEGY_BEST_METRIC` (`loss`, `accuracy`, `micro_precision`, `micro_recall`, `micro_f1`, `macro_precision`, `macro_recall`, `macro_f1`; default: `accuracy`)
- `CODLLM_MODEL_TASK` (`seq2seq`, `sequence_classification`; default: `seq2seq`)
- `CODLLM_LR_SCHEDULER_TYPE` (`linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `reduce_lr_on_plateau`; default: `linear`)
- `CODLLM_LR` (default: `3e-5`)
- `CODLLM_WEIGHT_DECAY` (default: `0.0`)
- `CODLLM_MAX_GRAD_NORM` (default: `0.5`)
- `CODLLM_TRAINING_INPUT` (comma-separated: `cod`, `age`, `sex`; default: `cod,age,sex`)
- `CODLLM_PRETRAIN_ENABLED` (`1`/`0`; when enabled, runs masterlist pretraining before normal training)
- `CODLLM_PRETRAIN_MASTERLIST_PATH` (default: `data/raw/ICD10h_Masterlist_2024.xlsx`)
- `CODLLM_PRETRAIN_MASTERLIST_SHEET_NAME` (default: `Masterlist`)
- `CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS` (default: `1`)
- `CODLLM_PRETRAIN_LEARNING_RATE` (optional; defaults to `CODLLM_LR` when unset)
- `CODLLM_PRETRAIN_LR_SCHEDULER_TYPE` (default: `linear`)
- `CODLLM_PRETRAIN_EVAL_EVERY_N_EPOCHS` (default: `1`)
- `CODLLM_PRETRAIN_UPSAMPLE_ENABLED` (`1`/`0`; default: `1`)
- `CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL` (default: `10`)
- `CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS` (comma-separated perturbations; default: `swap_adjacent_chars,delete_random_char,accent_random_vowel,qwerty_misspell`)
- `CODLLM_PRETRAIN_UPSAMPLE_PERTURBATIONS_PER_SAMPLE` (default: `1`)
- `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`
- `WANDB_DIR`, `WANDB_CACHE_DIR`, `XDG_CACHE_HOME_DIR`, `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`


## License

This repository is licensed under the MIT License.

Note that the historical datasets used for training and evaluation are not publicly available and
require separate agreements with the data providers.
