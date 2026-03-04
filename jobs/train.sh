#!/usr/bin/env bash

# ---------------- LSF directives ----------------
#BSUB -J codllm-train
#BSUB -q gpua100
#BSUB -W 04:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234805@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/%J.out
# -------------------------------------------------

set -euo pipefail

trap 'code=$?; printf "ERROR: jobs/train.sh failed at line %s with exit code %s\n" "$LINENO" "$code"; exit "$code"' ERR

PROJECT_DIR="${LSB_SUBCWD:-$(pwd)}"
cd "$PROJECT_DIR"
exec 2>&1

STORAGE_FOLDER="${STORAGE_FOLDER:-/work3/s234805}"
IMAGE_TAG="${IMAGE_TAG:-codllm-train:latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-dockerfiles/train.dockerfile}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
PULL_IMAGE="${PULL_IMAGE:-0}"
CONTAINER_NAME="codllm-train-${LSB_JOBID:-manual}"

if [ "${STORAGE_FOLDER#/}" = "$STORAGE_FOLDER" ]; then
  STORAGE_FOLDER="/$STORAGE_FOLDER"
fi

RUN_STORAGE_DIR="${RUN_STORAGE_DIR:-$STORAGE_FOLDER/codllm}"
HF_HOME="${HF_HOME:-$RUN_STORAGE_DIR/cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$RUN_STORAGE_DIR/cache/hf_datasets}"
TORCH_HOME="${TORCH_HOME:-$RUN_STORAGE_DIR/cache/torch}"
WANDB_DIR="${WANDB_DIR:-$RUN_STORAGE_DIR/cache/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
XDG_CACHE_HOME_DIR="${XDG_CACHE_HOME_DIR:-$RUN_STORAGE_DIR/cache/xdg}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$RUN_STORAGE_DIR/runs}"
TRAIN_DATA_RAW_DIR="${TRAIN_DATA_RAW_DIR:-$PROJECT_DIR/data/raw}"
TRAIN_DATA_PROCESSED_DIR="${TRAIN_DATA_PROCESSED_DIR:-$RUN_STORAGE_DIR/data/processed}"
CODLLM_SEED="${CODLLM_SEED:-42}"
PYTHONHASHSEED="${PYTHONHASHSEED:-$CODLLM_SEED}"
CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${LSB_DJOB_NUMPROC:-1}}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE TORCH_HOME
export WANDB_DIR WANDB_CACHE_DIR PYTHONUNBUFFERED=1
export CODLLM_SEED PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG
export OMP_NUM_THREADS MKL_NUM_THREADS TOKENIZERS_PARALLELISM

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
  "$TRAIN_OUTPUT_DIR" \
  "$TRAIN_DATA_PROCESSED_DIR"

if [ ! -d "$TRAIN_DATA_RAW_DIR" ]; then
  echo "ERROR: TRAIN_DATA_RAW_DIR '$TRAIN_DATA_RAW_DIR' does not exist."
  exit 1
fi

module load cuda/12.2

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not available in PATH on this host."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon is unavailable on this host."
  exit 1
fi

nvidia-smi

if [ "$BUILD_IMAGE" = "1" ]; then
  if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "ERROR: Dockerfile '$DOCKERFILE_PATH' does not exist."
    exit 1
  fi
  echo "Building image '$IMAGE_TAG' from '$DOCKERFILE_PATH'."
  docker build --file "$DOCKERFILE_PATH" --tag "$IMAGE_TAG" "$PROJECT_DIR"
elif docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Using existing local image '$IMAGE_TAG'."
elif [ "$PULL_IMAGE" = "1" ]; then
  echo "Pulling image '$IMAGE_TAG'."
  docker pull "$IMAGE_TAG"
  docker image inspect "$IMAGE_TAG" >/dev/null 2>&1
else
  echo "ERROR: Docker image '$IMAGE_TAG' is not available."
  echo "Set BUILD_IMAGE=1 to build it in the job, or set PULL_IMAGE=1 to pull it from a registry."
  exit 1
fi

docker_env_flags=(
  --env HF_HOME=/cache/huggingface
  --env HF_HUB_CACHE=/cache/huggingface/hub
  --env TRANSFORMERS_CACHE=/cache/huggingface/transformers
  --env HF_DATASETS_CACHE=/cache/hf_datasets
  --env TORCH_HOME=/cache/torch
  --env WANDB_DIR=/cache/wandb
  --env WANDB_CACHE_DIR=/cache/wandb/cache
  --env XDG_CACHE_HOME=/cache/xdg
  --env PYTHONUNBUFFERED=1
  --env CODLLM_SEED
  --env PYTHONHASHSEED
  --env CUBLAS_WORKSPACE_CONFIG
  --env OMP_NUM_THREADS
  --env MKL_NUM_THREADS
  --env TOKENIZERS_PARALLELISM
)

if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  docker_env_flags+=(--env HUGGINGFACE_HUB_TOKEN)
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  docker_env_flags+=(--env WANDB_API_KEY)
fi
if [ -n "${WANDB_MODE:-}" ]; then
  docker_env_flags+=(--env WANDB_MODE)
fi
if [ -n "${CODLLM_DATA_SEED:-}" ]; then
  docker_env_flags+=(--env CODLLM_DATA_SEED)
fi
if [ -n "${CODLLM_DATALOADER_NUM_WORKERS:-}" ]; then
  docker_env_flags+=(--env CODLLM_DATALOADER_NUM_WORKERS)
fi
if [ -n "${CODLLM_DETERMINISTIC_ALGORITHMS:-}" ]; then
  docker_env_flags+=(--env CODLLM_DETERMINISTIC_ALGORITHMS)
fi
if [ -n "${CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY:-}" ]; then
  docker_env_flags+=(--env CODLLM_DETERMINISTIC_ALGORITHMS_WARN_ONLY)
fi
if [ -n "${CODLLM_CUDNN_DETERMINISTIC:-}" ]; then
  docker_env_flags+=(--env CODLLM_CUDNN_DETERMINISTIC)
fi
if [ -n "${CODLLM_CUDNN_BENCHMARK:-}" ]; then
  docker_env_flags+=(--env CODLLM_CUDNN_BENCHMARK)
fi
if [ -n "${CODLLM_DATASET_SIZE:-}" ]; then
  docker_env_flags+=(--env CODLLM_DATASET_SIZE)
fi

echo "Starting training container '$CONTAINER_NAME' from image '$IMAGE_TAG'."
docker run \
  --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --mount type=bind,src="$HF_HOME",dst=/cache/huggingface \
  --mount type=bind,src="$HF_DATASETS_CACHE",dst=/cache/hf_datasets \
  --mount type=bind,src="$TORCH_HOME",dst=/cache/torch \
  --mount type=bind,src="$WANDB_DIR",dst=/cache/wandb \
  --mount type=bind,src="$XDG_CACHE_HOME_DIR",dst=/cache/xdg \
  --mount type=bind,src="$TRAIN_OUTPUT_DIR",dst=/app/runs \
  --mount type=bind,src="$TRAIN_DATA_RAW_DIR",dst=/app/data/raw,readonly \
  --mount type=bind,src="$TRAIN_DATA_PROCESSED_DIR",dst=/app/data/processed \
  "${docker_env_flags[@]}" \
  "$IMAGE_TAG"

echo "Training finished successfully."
