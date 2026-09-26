#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-pocketdiff_v4/runs/v440_3000_backbone_focused_20260925}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${PYTHONPATH:-.}"

exec conda run -n targetdiff torchrun --standalone --nproc_per_node=4 \
  --module pocketdiff_v4.train \
  --data pocketdiff_v4/data/residue_graphs_v42_contract.pt \
  --output-dir "${RUN_DIR}" \
  --epochs 200 \
  --updates 2400 \
  --batch-size 64 \
  --max-steps 2 \
  --train-steps 2 \
  --validation-steps 2 \
  --save-every 12 \
  --valid-every 1 \
  --log-every 24 \
  --num-workers 0 \
  --noise-scale-min 0.0 \
  --noise-scale-max 0.0 \
  --schedule-type remaining \
  --motion-scale 0.5 \
  --ca-motion-weight 0.1 \
  --ca-direction-weight 0.0 \
  --direction-weight 0.0 \
  --final-endpoint-weight 1.0 \
  --backbone-only-objective \
  --backbone-bridge-weight 1.0 \
  --backbone-endpoint-weight 1.0 \
  --bptt-steps 2 \
  --disable-chi
