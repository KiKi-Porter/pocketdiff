#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${PYTHONPATH:-.}"

exec conda run -n targetdiff torchrun --standalone --nproc_per_node=4 \
  --module pocketdiff_v4.train \
  --data pocketdiff_v4/data/residue_graphs_v42_contract.pt \
  --output-dir pocketdiff_v4/runs/v43_contract_3000_s4_20260924 \
  --epochs 200 \
  --updates 2400 \
  --batch-size 64 \
  --max-steps 4 \
  --validation-steps 4 \
  --save-every 12 \
  --valid-every 1 \
  --log-every 24 \
  --num-workers 0 \
  --noise-scale-min 0.0 \
  --noise-scale-max 0.0 \
  --disable-chi \
  --direction-weight 0.1
