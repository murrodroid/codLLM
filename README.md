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

- `jobs/train.sh` reads a fixed set of variables (listed below) and passes some into Docker.
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

## HPC Usage (LSF + Docker)

This repo already includes an HPC launcher: `jobs/train.sh`.

### 1) Edit scheduler directives in `jobs/train.sh`

Update the `#BSUB` lines for your cluster before first run:

- queue (`-q`)
- wall time (`-W`)
- CPU/GPU request (`-n`, `-gpu`)
- memory (`-R "rusage[mem=...]"`)
- email (`-u`)
- output log path (`-oo`)

### 2) Prepare job environment

From repo root:

```bash
mkdir -p logs

export STORAGE_FOLDER="/work3/$USER"               # set this to your own HPC storage
export HUGGINGFACE_HUB_TOKEN="YOUR_HF_KEY"         # optional, but recommended
export WANDB_API_KEY="YOUR_WANDB_KEY"              # optional
export CODLLM_SEED=42
export CODLLM_DATA_SEED=42

# First run on a node without image:
export BUILD_IMAGE=1
# Later runs:
export BUILD_IMAGE=0
```

`TRAIN_DATA_RAW_DIR` defaults to `data/raw`. Make sure the expected raw files are present there
(or override the path):

- `SOSA_EXTR_1920-1930 (belgium).xlsx`
- `AMC_1854_1926_LM.csv`

### 3) Submit

```bash
bsub < jobs/train.sh
```

### 4) Monitor

```bash
bjobs
tail -f logs/<job_id>.out
```

### 5) Outputs

By default, results and caches are written under:

- `${RUN_STORAGE_DIR:-$STORAGE_FOLDER/codllm}/runs` (training outputs)
- `${RUN_STORAGE_DIR:-$STORAGE_FOLDER/codllm}/data/processed` (processed data)
- `${RUN_STORAGE_DIR:-$STORAGE_FOLDER/codllm}/cache/*` (HF/Torch/W&B caches)

Processed-data caching is setup-aware:

- The pipeline writes a sidecar metadata file (`<processed-file>.meta.json`).
- On the next run, processed data is reused only if metadata still matches the current
  data setup (sources, mappings, and relevant preprocessing config).
- If anything changes, processed data is rebuilt automatically.

### HPC env vars supported by `jobs/train.sh`

- `STORAGE_FOLDER` (default: `/work3/s234805`)
- `RUN_STORAGE_DIR` (default: `$STORAGE_FOLDER/codllm`)
- `IMAGE_TAG` (default: `codllm-train:latest`)
- `DOCKERFILE_PATH` (default: `dockerfiles/train.dockerfile`)
- `BUILD_IMAGE` (`1` to build in job, default `0`)
- `PULL_IMAGE` (`1` to pull image if missing, default `0`)
- `TRAIN_DATA_RAW_DIR` (default: `$PROJECT_DIR/data/raw`)
- `TRAIN_DATA_PROCESSED_DIR` (default: `$RUN_STORAGE_DIR/data/processed`)
- `TRAIN_OUTPUT_DIR` (default: `$RUN_STORAGE_DIR/runs`)
- `HUGGINGFACE_HUB_TOKEN` (forwarded to container if set)
- `WANDB_API_KEY` (forwarded to container if set)
- `WANDB_MODE` (forwarded to container if set)
- `CODLLM_SEED` (default: `42`)
- `CODLLM_DATA_SEED` (defaults to `CODLLM_SEED` when unset)
- `CODLLM_DATALOADER_NUM_WORKERS` (default: `0`)
- `CODLLM_DETERMINISTIC_ALGORITHMS` (optional override)
- `CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY` (optional override)
- `CODLLM_CUDNN_DETERMINISTIC` (optional override)
- `CODLLM_CUDNN_BENCHMARK` (optional override)
- `CODLLM_DATASET_SIZE` (optional override)
- `CODLLM_MAX_LABEL_COUNT` (optional override)
- `CODLLM_MAX_TARGET_LENGTH` (optional override)
- `CODLLM_LABEL_CODE_LENGTH` (optional override, default `7`)
- `CODLLM_LABEL_SEPARATOR` (optional override, default `" | "`)
- `CODLLM_MAX_TARGET_LENGTH_BUFFER` (optional override, default `4`)
- `PYTHONHASHSEED` (default: `CODLLM_SEED`)
- `CUBLAS_WORKSPACE_CONFIG` (default: `:4096:8`)
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM` (job defaults set)
- `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`,
  `WANDB_DIR`, `WANDB_CACHE_DIR`, `XDG_CACHE_HOME_DIR` (optional cache overrides)

## License

This repository is licensed under the MIT License.

Note that the historical datasets used for training and evaluation are not publicly available and
require separate agreements with the data providers.
