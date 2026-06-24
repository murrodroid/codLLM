#!/usr/bin/env bash
# STEP 3: run inference per source, each LOSO checkpoint scoring ITS OWN held-out source.
# Single-writer per output file; sequential (one GPU).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
PY=./.venv-gpu/Scripts/python.exe
for SRC in amsterdam copenhagen madrid belgium ipswich; do
  echo "=== INFER $SRC ==="
  "$PY" -m experiments.uncertainty.local_infer \
    --checkpoint "experiments/uncertainty/results/loso/ckpt_${SRC}" \
    --texts "experiments/uncertainty/results/loso/texts_${SRC}.jsonl" \
    --output "experiments/uncertainty/results/loso/${SRC}_holdout.jsonl" \
    --batch-size 64 --log-every 5000
done
echo "INFER ALL DONE"
