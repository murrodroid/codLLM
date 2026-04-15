#!/usr/bin/env bash

set -euo pipefail

trap 'code=$?; printf "ERROR: jobs/inference.sh failed at line %s with exit code %s\n" "$LINENO" "$code"; exit "$code"' ERR

PROJECT_DIR="${LSB_SUBCWD:-$(pwd)}"
cd "$PROJECT_DIR"
exec 2>&1

JOB_CONFIG_FILE="${JOB_CONFIG_FILE:-${1:-}}"
if [ -n "$JOB_CONFIG_FILE" ]; then
  if [ -f "$JOB_CONFIG_FILE" ]; then
    resolved_job_config="$JOB_CONFIG_FILE"
  elif [ -f "$PROJECT_DIR/$JOB_CONFIG_FILE" ]; then
    resolved_job_config="$PROJECT_DIR/$JOB_CONFIG_FILE"
  elif [ -f "$PROJECT_DIR/jobs/configs/$JOB_CONFIG_FILE" ]; then
    resolved_job_config="$PROJECT_DIR/jobs/configs/$JOB_CONFIG_FILE"
  else
    echo "ERROR: JOB_CONFIG_FILE '$JOB_CONFIG_FILE' does not exist."
    exit 1
  fi
  resolved_job_config="$(cd "$(dirname "$resolved_job_config")" && pwd)/$(basename "$resolved_job_config")"
  echo "Loading inference config file: $resolved_job_config"
  set -a
  source "$resolved_job_config"
  set +a
fi

INFERENCE_INPUT_PATH="${INFERENCE_INPUT_PATH:-}"
INFERENCE_OUTPUT_PATH="${INFERENCE_OUTPUT_PATH:-}"
CODLLM_INFERENCE_VALIDATE_REGISTRY="${CODLLM_INFERENCE_VALIDATE_REGISTRY:-0}"

if [ -z "$INFERENCE_INPUT_PATH" ]; then
  echo "ERROR: INFERENCE_INPUT_PATH must be set."
  exit 1
fi

inference_cmd=(uv run python -m codllm.inference "$INFERENCE_INPUT_PATH")
if [ -n "$INFERENCE_OUTPUT_PATH" ]; then
  inference_cmd+=(--output-path "$INFERENCE_OUTPUT_PATH")
fi
if [ "$CODLLM_INFERENCE_VALIDATE_REGISTRY" = "1" ] || [ "$CODLLM_INFERENCE_VALIDATE_REGISTRY" = "true" ]; then
  inference_cmd+=(--validate-registry)
fi

echo "Running inference command: ${inference_cmd[*]}"
"${inference_cmd[@]}"
