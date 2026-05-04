# Coding Historical Causes of Death with Large Language Models

codLLM trains and evaluates transformer models for mapping historical free-text causes of death to ICD10h labels. The
main runtime is the `codllm` Python package under `src/`, with entrypoints for training and inference, TOML experiment
specs under `runs/`, and invoke tasks for local and LSF/HPC workflows.

![codLLM training pipeline](visualizations/codLLM_pipeline.png)

## Project Scope

Historical cause-of-death coding is usually manual, slow, and dependent on specialist knowledge. codLLM makes this work
repeatable by standardizing source datasets, harmonizing ICD10h labels, preparing train/validation/test and hold-out
splits, fine-tuning Hugging Face models, and logging model behavior and data diagnostics.

The repository does not include the historical datasets. Raw data and the ICD10h masterlist must be supplied separately
under `data/raw/` or through `CODLLM_DATA_RAW_DIR`.

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

Set credentials in your shell environment:

```bash
export HUGGINGFACE_HUB_TOKEN="..."
export WANDB_API_KEY="..."
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
uv run invoke train --config runs/single/base_small.toml
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

Processed-data caches are validated against source file signatures, source mappings, label harmonization settings, input
fields, and label settings. Prepared split caches live beside the processed file under:

```text
<processed-stem>.splits/<cache-key>/
```

Those split caches include split-time transformations such as hold-out removal, multi-COD synthesis, balancing, and
masterlist injection.

## Repository Layout

```text
codLLM/
|-- AGENTS.md                       # Agent instructions and project conventions
|-- README.md
|-- pyproject.toml                  # Python package metadata and dependencies
|-- uv.lock
|-- tasks.py                        # Invoke tasks for local, experiment, and HPC workflows
|-- src/codllm/
|   |-- training/                   # Training CLI, stages, Trainer setup, W&B logging
|   |-- inference/                  # Inference CLI, IO, generation, decoding
|   |-- settings/                   # Config dataclasses, env parsing, option types
|   |-- input/                      # Source mappings, raw loaders, harmonization, multi-COD
|   |-- data/                       # DataHandler, caches, splits, balancing, tokenization
|   |-- experiments/                # TOML specs and LSF submission rendering
|   |-- labels/                     # ICD10h schemas and registry helpers
|   |-- models/                     # Hugging Face model loading
|   `-- runtime/                    # Paths and reproducibility helpers
|-- runs/
|   |-- base/                       # Reusable H100 runtime bases
|   |-- single/                     # Single-run experiment specs
|   `-- sweeps/                     # Cartesian sweep specs
|-- hpc/
|   |-- env.sh                      # HPC storage/cache bootstrap
|   |-- lsf_profiles.toml           # Named LSF resource profiles
|   `-- storage-check.sh
|-- data/
|   |-- raw/                        # User-supplied raw data, not committed
|   `-- processed/                  # Local processed caches, not committed
|-- visualizations/                 # README and project figures
|-- experiments/                    # Notebooks and tree-search experiments
|-- dockerfiles/
|   `-- train.dockerfile
|-- tests/
`-- jobs/generated/                 # Created by hpc.submit; ignored by git
```

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
one run per cartesian combination. Sweep runs also get generated W&B grouping metadata unless explicitly overridden:
`WANDB_SWEEP_ID=codllm-<experiment-name>` and `WANDB_RUN_GROUP=<experiment-name>`.

Useful commands:

```bash
uv run invoke --list
uv run invoke experiments.list
uv run invoke experiments.plan --config runs/sweeps/pretraining.toml --profile h100-10h
```

Checked-in specs:

| Spec | Purpose |
| --- | --- |
| `runs/base/h100-small.toml` | H100 runtime base for `google/flan-t5-small` |
| `runs/base/h100-large.toml` | H100 runtime base for `google/flan-t5-large` |
| `runs/single/base_small.toml` | Current single-run baseline with multi-COD, balancing, and pretraining enabled |
| `runs/sweeps/pretraining.toml` | Masterlist pretraining on/off |
| `runs/sweeps/multicod_pretrain.toml` | Synthetic multi-COD masterlist pretraining ratio |
| `runs/sweeps/multicod.toml` | Multi-COD synthetic fine-tuning ratio with the large H100 base |
| `runs/sweeps/holdout.toml` | Leave-one-source-out evaluation |
| `runs/sweeps/scheduler.toml` | LR scheduler comparison |
| `runs/sweeps/training_inputs.toml` | COD-only input vs COD+age+sex input |

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

The default profiles are `h100-2h` through `h100-10h`, `h100-24h`, and `v100`. H100 profiles request 17 CPU cores so
training specs can use 16 DataLoader workers plus the main process.

Inspect a concrete experiment plan:

```bash
uv run --no-sync invoke experiments.plan \
  --config runs/single/base_small.toml \
  --profile h100-10h
```

Prebuild reusable processed-data and prepared-split caches:

```bash
uv run --no-sync invoke hpc.build \
  --config runs/single/base_small.toml \
  --profile h100-10h
```

For sweep specs, `hpc.build` builds every expanded run by default. Build one run with a one-based sweep index:

```bash
uv run --no-sync invoke hpc.build \
  --config runs/sweeps/multicod_pretrain.toml \
  --profile h100-10h \
  --sweep-index 2
```

Generate an LSF job without submitting it:

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

`--user` currently supports `lucas` and `elias` for LSF notification email selection. Sweep specs render as LSF job
arrays with one generated env file per array index. The generated job script sets storage/cache defaults, loads profile
modules, runs `uv sync --frozen --no-dev` under a lock unless `SYNC_ENV=0`, and launches either training or inference
from the generated run environment.

HPC defaults:

- raw data: `$PROJECT_DIR/data/raw`
- processed data: `$RUN_STORAGE_DIR/data/processed`
- model outputs: `$RUN_STORAGE_DIR/runs`
- generated scripts/env files/manifests: `jobs/generated/`
- logs: `logs/`

## Training Options

All runtime behavior is owned by `Config` in `src/codllm/settings/schema.py` and environment overrides in
`src/codllm/settings/env.py`. Prefer setting options in TOML specs rather than editing code.

### Model Size and Task

- `CODLLM_HF_MODEL` selects a Hugging Face model id or local checkpoint path.
- `runs/base/h100-small.toml` uses `google/flan-t5-small`.
- `runs/base/h100-large.toml` uses `google/flan-t5-large`.
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

Controls:

- `CODLLM_BALANCE_STRATEGY=none|upsample|sqrt`
- `CODLLM_BALANCE_TARGET_QUANTILE`
- `CODLLM_BALANCE_UPSAMPLE_LABELS`
- `CODLLM_BALANCE_UPSAMPLE_INVERSE_POWER`
- `CODLLM_BALANCE_UPSAMPLE_BUDGET_RATIO`
- `CODLLM_BALANCE_SQRT_FLOOR`
- `CODLLM_BALANCE_SQRT_DECAY`
- `CODLLM_BALANCE_SQRT_POWER`
- `CODLLM_BALANCE_SQRT_BUDGET_SCALE`
- `CODLLM_BALANCE_PERTURBATIONS`
- `CODLLM_BALANCE_BASE_PERTURBATION_RATE`
- `CODLLM_BALANCE_PERTURBATION_MEAN`
- `CODLLM_BALANCE_PERTURBATION_VARIANCE`

Perturbations are applied only to the configured `cod` segment. The mean and variance scale with COD text length, so
longer cause strings receive more edits on average.

### Hold-Out Evaluation

Set `CODLLM_HOLD_OUT_DATASET` to a processed `source_id` for leave-one-source-out evaluation. Matching rows are removed
before dataset sampling and train/validation/test splitting. After training, the full held-out source is evaluated with
`holdout_test` metrics.

During-training hold-out evaluation is optional:

- `CODLLM_HOLD_OUT_EVALUATE_PER=epoch|steps`
- `CODLLM_HOLD_OUT_EVALUATE_RATIO=0.05`

The during-training callback evaluates a deterministic sample of the held-out source. The final post-training hold-out
evaluation always uses the full held-out source.

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

## Outputs and Metrics

Each training run writes checkpoints under a run-scoped output directory. Locally this becomes `runs/run-0001`,
`runs/run-0002`, and so on. On LSF, run ids use `LSB_JOBID` and `LSB_JOBINDEX`.

When W&B credentials are available, training logs:

- resolved config and runtime metadata
- source file metadata and processed-cache fingerprints
- split sizes, source distributions, and label summaries
- compact data visualizations under `data/*`
- validation, test, hold-out, and pretraining metrics under scoped namespaces
- error tables under scopes such as `val/errors/*`, `test/errors/*`, and `holdout/test/errors/*`

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
