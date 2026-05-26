#!/usr/bin/env bash

# Source this file before running uv/invoke commands on HPC login nodes.
# It must run before `uv run ...`, because uv chooses its cache and venv paths
# before Python task code starts.

: "${STORAGE_FOLDER:=/work3/${USER:?USER must be set or STORAGE_FOLDER must be provided}}"
: "${RUN_STORAGE_DIR:=$STORAGE_FOLDER/codllm}"

export STORAGE_FOLDER
export RUN_STORAGE_DIR

export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN_STORAGE_DIR/cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$RUN_STORAGE_DIR/.venv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RUN_STORAGE_DIR/python}"

export HF_HOME="${HF_HOME:-$RUN_STORAGE_DIR/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$RUN_STORAGE_DIR/cache/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$RUN_STORAGE_DIR/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-$RUN_STORAGE_DIR/cache/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_STORAGE_DIR/cache/xdg}"

# Reduce CUDA allocator fragmentation. flan-t5-xl sits near the 80 GB H100
# ceiling at the pretrain->finetune transfer; expandable_segments lets the
# allocator reclaim reserved-but-unallocated blocks instead of OOMing on a
# tiny allocation amid fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "$RUN_STORAGE_DIR" \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$XDG_CACHE_HOME"

mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

printf 'Configured codLLM HPC storage:\n'
printf '  RUN_STORAGE_DIR=%s\n' "$RUN_STORAGE_DIR"
printf '  UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
printf '  UV_PROJECT_ENVIRONMENT=%s\n' "$UV_PROJECT_ENVIRONMENT"
printf '  UV_PYTHON_INSTALL_DIR=%s\n' "$UV_PYTHON_INSTALL_DIR"
