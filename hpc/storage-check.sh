#!/usr/bin/env bash

set -euo pipefail

printf 'codLLM storage environment:\n'
printf '  RUN_STORAGE_DIR=%s\n' "${RUN_STORAGE_DIR:-<unset>}"
printf '  UV_CACHE_DIR=%s\n' "${UV_CACHE_DIR:-<unset>}"
printf '  UV_PROJECT_ENVIRONMENT=%s\n' "${UV_PROJECT_ENVIRONMENT:-<unset>}"
printf '  UV_PYTHON_INSTALL_DIR=%s\n' "${UV_PYTHON_INSTALL_DIR:-<unset>}"
printf '  HF_HOME=%s\n' "${HF_HOME:-<unset>}"
printf '  HF_HUB_CACHE=%s\n' "${HF_HUB_CACHE:-<unset>}"
printf '  TRANSFORMERS_CACHE=%s\n' "${TRANSFORMERS_CACHE:-<unset>}"
printf '  HF_DATASETS_CACHE=%s\n' "${HF_DATASETS_CACHE:-<unset>}"
printf '  TORCH_HOME=%s\n' "${TORCH_HOME:-<unset>}"
printf '  WANDB_DIR=%s\n' "${WANDB_DIR:-<unset>}"
printf '  WANDB_CACHE_DIR=%s\n' "${WANDB_CACHE_DIR:-<unset>}"
printf '  XDG_CACHE_HOME=%s\n' "${XDG_CACHE_HOME:-<unset>}"
printf '  VIRTUAL_ENV=%s\n' "${VIRTUAL_ENV:-<unset>}"

if command -v uv >/dev/null 2>&1; then
  printf '  uv cache dir=%s\n' "$(uv cache dir)"
else
  printf '  uv cache dir=<uv not found>\n'
fi

if [ -z "${RUN_STORAGE_DIR:-}" ]; then
  printf 'WARNING: RUN_STORAGE_DIR is unset. Source hpc/env.sh before running uv on HPC.\n'
  exit 1
fi

warn_if_outside_storage() {
  local name="$1"
  local value="${2:-}"
  case "$value" in
    "")
      return 0
      ;;
    "$RUN_STORAGE_DIR"|"$RUN_STORAGE_DIR"/*)
      return 0
      ;;
    *)
      printf 'WARNING: %s is outside RUN_STORAGE_DIR: %s\n' "$name" "$value"
      ;;
  esac
}

warn_if_outside_storage UV_CACHE_DIR "${UV_CACHE_DIR:-}"
warn_if_outside_storage UV_PROJECT_ENVIRONMENT "${UV_PROJECT_ENVIRONMENT:-}"
warn_if_outside_storage UV_PYTHON_INSTALL_DIR "${UV_PYTHON_INSTALL_DIR:-}"
warn_if_outside_storage HF_HOME "${HF_HOME:-}"
warn_if_outside_storage HF_HUB_CACHE "${HF_HUB_CACHE:-}"
warn_if_outside_storage TRANSFORMERS_CACHE "${TRANSFORMERS_CACHE:-}"
warn_if_outside_storage HF_DATASETS_CACHE "${HF_DATASETS_CACHE:-}"
warn_if_outside_storage TORCH_HOME "${TORCH_HOME:-}"
warn_if_outside_storage WANDB_DIR "${WANDB_DIR:-}"
warn_if_outside_storage WANDB_CACHE_DIR "${WANDB_CACHE_DIR:-}"
warn_if_outside_storage XDG_CACHE_HOME "${XDG_CACHE_HOME:-}"
