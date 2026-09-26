#!/usr/bin/env bash
set -euo pipefail

DATA="pocketdiff_v4/data/residue_graphs_v41.pt"
CHECKPOINT="${1:?usage: run_v41_evaluate_controls.sh CHECKPOINT OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: run_v41_evaluate_controls.sh CHECKPOINT OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

for spec in \
  apo_off:0.0:0.0 \
  noisy_off:1.0:0.0 \
  apo_learned:0.0:1.0 \
  noisy_learned:1.0:1.0
do
  IFS=: read -r name noise motion <<< "${spec}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" PYTHONPATH=. \
    conda run -n targetdiff torchrun --standalone --nproc_per_node=1 \
    --module pocketdiff_v4.evaluate \
    --data "${DATA}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT_DIR}/${name}.json" \
    --batch-size 16 \
    --max-nodes 12000 \
    --steps 4 \
    --seed 20260924 \
    --motion-scale "${motion}" \
    --initial-noise-scale "${noise}"
done
