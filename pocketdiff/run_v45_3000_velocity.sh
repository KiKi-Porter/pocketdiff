#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-pocketdiff/runs/v45_3000_velocity}"
SOURCE_CACHE="${SOURCE_CACHE:-pocketdiff_v1/data/apo2mol_3000_cache.pt}"
DEVICE="${DEVICE:-cuda}"

PYTHONPATH="$ROOT" python -m pocketdiff.scripts.train_diffusion \
  --output-dir "$OUTPUT_DIR" \
  --source-cache "$SOURCE_CACHE" \
  --device "$DEVICE" \
  --train-count 3000 \
  --valid-count 300 \
  --test-count 300 \
  --holdout-count 300 \
  --updates 2000 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 \
  --sampler-steps 20 \
  --prediction-type velocity \
  --time-min 0.02 \
  --time-max 1.0 \
  --translation-noise-scale 0.05 \
  --rotation-noise-scale 0.025 \
  --chi-noise-scale 0.025 \
  --endpoint-weight 1.0 \
  --backbone-endpoint-weight 2.0 \
  --continuity-weight 0.10 \
  --direction-weight 0.05 \
  --direction-threshold 0.05 \
  --motion-bucket-count 4

PYTHONPATH="$ROOT" python -m pocketdiff.scripts.evaluate_diffusion \
  --checkpoint "$OUTPUT_DIR/checkpoint.pt" \
  --source-cache "$SOURCE_CACHE" \
  --device "$DEVICE" \
  --valid-count 300 \
  --test-count 300 \
  --steps 20,50,100 \
  --output "$OUTPUT_DIR/evaluation_20_50_100.json"
