#!/usr/bin/env bash

# ---------------- LSF directives ----------------
#BSUB -J codllm-train
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
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

PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

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

CODLLM_SEED="${CODLLM_SEED:-42}"
PYTHONHASHSEED="${PYTHONHASHSEED:-$CODLLM_SEED}"
CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${LSB_DJOB_NUMPROC:-1}}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
UV_BIN="${UV_BIN:-uv}"
UV_CMD=()

export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE TORCH_HOME
export WANDB_DIR WANDB_CACHE_DIR
export XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT
export CODLLM_OUTPUT_DIR CODLLM_DATA_RAW_DIR CODLLM_DATA_PROCESSED_DIR
export CODLLM_SEED PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG
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

if [ -n "${UV_BIN:-}" ] && [ -x "$UV_BIN" ]; then
  UV_CMD=("$UV_BIN")
elif command -v "$UV_BIN" >/dev/null 2>&1; then
  UV_CMD=("$(command -v "$UV_BIN")")
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_CMD=("$HOME/.local/bin/uv")
elif [ -x "$HOME/.cargo/bin/uv" ]; then
  UV_CMD=("$HOME/.cargo/bin/uv")
else
  for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -m uv --version >/dev/null 2>&1; then
      UV_CMD=("$py" "-m" "uv")
      break
    fi
  done
fi

if [ "${#UV_CMD[@]}" -eq 0 ]; then
  echo "ERROR: uv is not available in PATH."
  echo "Tried UV_BIN='$UV_BIN', \$HOME/.local/bin/uv, \$HOME/.cargo/bin/uv, and python -m uv."
  echo "HOME='${HOME:-<unset>}'"
  echo "PATH='$PATH'"
  echo "Set UV_BIN to the full uv path, e.g. export UV_BIN=\$HOME/.local/bin/uv"
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
  "${UV_CMD[@]}" sync --frozen --no-dev
fi

train_cmd=("${UV_CMD[@]}" run python -m codllm.train)
if [ "$FORCE_REPROCESS" = "1" ]; then
  train_cmd+=(--force-reprocess)
fi

if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
  read -r -a extra_args <<< "$TRAIN_EXTRA_ARGS"
  train_cmd+=("${extra_args[@]}")
fi

echo "Starting native training on host with command: ${train_cmd[*]}"
"${train_cmd[@]}"

echo "Training finished successfully."
