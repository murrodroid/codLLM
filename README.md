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

## Runtime Environment Variables

Environment variables are supported, but only if they are explicitly read by the scripts.

- `jobs/train.sh` reads a fixed set of variables and runs on HPC.
- The Python training code reads `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) and `WANDB_API_KEY`.
- Reproducibility controls are read from `CODLLM_*` env vars (listed below).
- Arbitrary env vars are ignored unless the code references them.

## Reproducibility Defaults

The training runtime uses config-driven reproducibility defaults:

- global seed (`seed`)
- data split/sampler seed (`data_seed`, defaults to `seed`)
- deterministic torch algorithms
- deterministic CuDNN mode
- fixed dataloader worker count (`0` by default)
- dynamic target-length floor for labels

You can override these at runtime without editing code:

```bash
export CODLLM_SEED=42
export CODLLM_DATA_SEED=42
export CODLLM_DATALOADER_NUM_WORKERS=0
export CODLLM_DETERMINISTIC_ALGORITHMS=true
export CODLLM_CUDNN_DETERMINISTIC=true
export CODLLM_CUDNN_BENCHMARK=false
export CODLLM_LOAD_IN_8BIT=0
export CODLLM_VERBOSE=true
export CODLLM_TORCH_DTYPE=auto
export CODLLM_WANDB_LOG_MODEL=end
export CODLLM_WARMUP_STEPS=1000
export CODLLM_NUM_TRAIN_EPOCHS=4
export CODLLM_LR=3e-5
export CODLLM_WEIGHT_DECAY=0.0
export CODLLM_MAX_GRAD_NORM=0.5
export CODLLM_TRAINING_INPUT="cod,age,sex"
export CODLLM_MAX_LABEL_COUNT=2
export CODLLM_MAX_TARGET_LENGTH=16
export CODLLM_LABEL_CODE_LENGTH=7
export CODLLM_LABEL_SEPARATOR=" | "
export CODLLM_MAX_TARGET_LENGTH_BUFFER=4
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

## HPC Usage (LSF, No Docker)

Use this path when your cluster does not allow Docker.

Script: `jobs/train.sh`

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

LSF inherits exported environment variables from the submitting shell, so set them before
`bsub`.

You can also submit using a job config file:

```bash
JOB_CONFIG_FILE=jobs/configs/example.env bsub < jobs/train.sh
```

`JOB_CONFIG_FILE` accepts simple shell `KEY=value` lines (comments with `#` are allowed).
See `jobs/configs/example.env`.

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
- `CODLLM_DATA_RAW_DIR`, `CODLLM_DATA_PROCESSED_DIR`, `CODLLM_OUTPUT_DIR` (optional overrides)
- `SYNC_ENV` (`1` to run `uv sync`, default `1`)
- `UV_SYNC_LOCK_FILE` (lock file used to serialize `uv sync`, default: `$RUN_STORAGE_DIR/.uv-sync.lock`)
- `FORCE_REPROCESS` (`1` adds `--force-reprocess`, default `0`)
- `TRAIN_EXTRA_ARGS` (optional args appended to `python -m codllm.train`)
- `HUGGINGFACE_HUB_TOKEN`, `WANDB_API_KEY`, `WANDB_MODE`
- `CODLLM_*` training/reproducibility settings from the section above
- `CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS` (processed-cache lock wait timeout, default: `900`)
- `CODLLM_RUN_DIR_LOCK_TIMEOUT_SECONDS` (run-dir lock wait timeout, default: `120`)
- `CODLLM_LOAD_IN_8BIT` (`0` by default in `jobs/train.sh`)
- `CODLLM_VERBOSE` (`1`/`0`, default: `1`; prints resolved setup before training)
- `CODLLM_TORCH_DTYPE` (`auto`, `float16`, `bfloat16`, `float32`)
- `CODLLM_WANDB_LOG_MODEL` (`false`, `end`, `checkpoint`; default: `end`)
- `CODLLM_WARMUP_STEPS` (default: `1000`)
- `CODLLM_NUM_TRAIN_EPOCHS` (default: `4`)
- `CODLLM_LR` (default: `3e-5`)
- `CODLLM_WEIGHT_DECAY` (default: `0.0`)
- `CODLLM_MAX_GRAD_NORM` (default: `0.5`)
- `CODLLM_TRAINING_INPUT` (comma-separated: `cod`, `age`, `sex`; default: `cod,age,sex`)
- `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`
- `WANDB_DIR`, `WANDB_CACHE_DIR`, `XDG_CACHE_HOME_DIR`, `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`


## License

This repository is licensed under the MIT License.

Note that the historical datasets used for training and evaluation are not publicly available and
require separate agreements with the data providers.
