#!/usr/bin/env bash
# ---------------- LSF directives ----------------
#BSUB -J codllm-train
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234805@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/%J.out
# -------------------------------------------------

set -euo pipefail

trap 'code=$?;
  printf "ERROR: jobs/train.sh failed at line %s with exit code %s\n" "$LINENO" "$code";
  exit "$code"' ERR

PROJECT_DIR="${LSB_SUBCWD:-$(pwd)}"
cd "$PROJECT_DIR"
exec 2>&1

JOB_CONFIG_FILE="${JOB_CONFIG_FILE:-}"
if [ -n "$JOB_CONFIG_FILE" ]; then
  if [ -f "$JOB_CONFIG_FILE" ]; then
    resolved_job_config="$JOB_CONFIG_FILE"
  elif [ -f "$PROJECT_DIR/$JOB_CONFIG_FILE" ]; then
    resolved_job_config="$PROJECT_DIR/$JOB_CONFIG_FILE"
  else
    echo "ERROR: JOB_CONFIG_FILE '$JOB_CONFIG_FILE' does not exist."
    exit 1
  fi
  echo "Loading job config file: $resolved_job_config"
  set -a
  source "$resolved_job_config"
  set +a
fi

STORAGE_FOLDER="${STORAGE_FOLDER:-/work3/s234805}"
RUN_STORAGE_DIR="${RUN_STORAGE_DIR:-$STORAGE_FOLDER/codllm}"

HF_HOME="${HF_HOME:-$RUN_STORAGE_DIR/cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$RUN_STORAGE_DIR/cache/hf_datasets}"
TORCH_HOME="${TORCH_HOME:-$RUN_STORAGE_DIR/cache/torch}"
WANDB_DIR="${WANDB_DIR:-$RUN_STORAGE_DIR/cache/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
XDG_CACHE_HOME_DIR="${XDG_CACHE_HOME_DIR:-$RUN_STORAGE_DIR/cache/xdg}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN_STORAGE_DIR/cache/uv}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$RUN_STORAGE_DIR/.venv}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$RUN_STORAGE_DIR/runs}"
TRAIN_DATA_RAW_DIR="${TRAIN_DATA_RAW_DIR:-$PROJECT_DIR/data/raw}"
TRAIN_DATA_PROCESSED_DIR="${TRAIN_DATA_PROCESSED_DIR:-$RUN_STORAGE_DIR/data/processed}"

CODLLM_OUTPUT_DIR="${CODLLM_OUTPUT_DIR:-$TRAIN_OUTPUT_DIR}"
CODLLM_DATA_RAW_DIR="${CODLLM_DATA_RAW_DIR:-$TRAIN_DATA_RAW_DIR}"
CODLLM_DATA_PROCESSED_DIR="${CODLLM_DATA_PROCESSED_DIR:-$TRAIN_DATA_PROCESSED_DIR}"

FORCE_REPROCESS="${FORCE_REPROCESS:-0}"
SYNC_ENV="${SYNC_ENV:-1}"
UV_SYNC_LOCK_FILE="${UV_SYNC_LOCK_FILE:-$RUN_STORAGE_DIR/.uv-sync.lock}"

CODLLM_SEED="${CODLLM_SEED:-42}"
CODLLM_LOAD_IN_8BIT="${CODLLM_LOAD_IN_8BIT:-0}"
CODLLM_TORCH_DTYPE="${CODLLM_TORCH_DTYPE:-auto}"
CODLLM_WANDB_LOG_MODEL="${CODLLM_WANDB_LOG_MODEL:-end}"
CODLLM_WARMUP_STEPS="${CODLLM_WARMUP_STEPS:-1000}"
CODLLM_NUM_TRAIN_EPOCHS="${CODLLM_NUM_TRAIN_EPOCHS:-4}"
CODLLM_LR="${CODLLM_LR:-3e-5}"
CODLLM_WEIGHT_DECAY="${CODLLM_WEIGHT_DECAY:-0.0}"
CODLLM_MAX_GRAD_NORM="${CODLLM_MAX_GRAD_NORM:-0.5}"
CODLLM_TRAINING_INPUT="${CODLLM_TRAINING_INPUT:-cod,age,sex}"
PYTHONHASHSEED="${PYTHONHASHSEED:-$CODLLM_SEED}"
CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${LSB_DJOB_NUMPROC:-1}}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE TORCH_HOME
export WANDB_DIR WANDB_CACHE_DIR
export XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT
export CODLLM_OUTPUT_DIR CODLLM_DATA_RAW_DIR CODLLM_DATA_PROCESSED_DIR
export CODLLM_SEED CODLLM_LOAD_IN_8BIT CODLLM_TORCH_DTYPE
export CODLLM_WANDB_LOG_MODEL
export CODLLM_WARMUP_STEPS CODLLM_NUM_TRAIN_EPOCHS
export CODLLM_LR CODLLM_WEIGHT_DECAY CODLLM_MAX_GRAD_NORM
export CODLLM_TRAINING_INPUT
export PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG
export OMP_NUM_THREADS MKL_NUM_THREADS TOKENIZERS_PARALLELISM
export PYTHONUNBUFFERED=1

mkdir -p \
  "$RUN_STORAGE_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$XDG_CACHE_HOME_DIR" \
  "$UV_CACHE_DIR" \
  "$CODLLM_OUTPUT_DIR" \
  "$CODLLM_DATA_PROCESSED_DIR"

if [ ! -d "$CODLLM_DATA_RAW_DIR" ]; then
  echo "ERROR: CODLLM_DATA_RAW_DIR '$CODLLM_DATA_RAW_DIR' does not exist."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available in PATH."
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

if [ "$SYNC_ENV" = "1" ]; then
  echo "Syncing Python environment with uv."
  if command -v flock >/dev/null 2>&1; then
    flock "$UV_SYNC_LOCK_FILE" uv sync --frozen --no-dev
  else
    echo "WARNING: flock is not available; running uv sync without cross-job locking."
    uv sync --frozen --no-dev
  fi
fi

train_cmd=(uv run python -m codllm.train)
if [ "$FORCE_REPROCESS" = "1" ]; then
  train_cmd+=(--force-reprocess)
fi

if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
  read -r -a extra_args <<< "$TRAIN_EXTRA_ARGS"
  train_cmd+=("${extra_args[@]}")
fi

echo "Starting training on host with command: ${train_cmd[*]}"
"${train_cmd[@]}"

echo "Training finished successfully."
