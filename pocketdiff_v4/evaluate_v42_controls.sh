#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-pocketdiff_v4/runs/v43_contract_3000_s4_20260924}"
CHECKPOINT="${2:-${RUN_DIR}/best.pt}"
OUT_DIR="${3:-${RUN_DIR}/controls_best}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${PYTHONPATH:-.}"

mkdir -p "${OUT_DIR}"

run_eval() {
  local name="$1"
  local motion_scale="$2"
  local initial_noise_scale="$3"
  conda run -n targetdiff torchrun --standalone --nproc_per_node=4 \
    --module pocketdiff_v4.evaluate \
    --data pocketdiff_v4/data/residue_graphs_v42_contract.pt \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUT_DIR}/${name}.json" \
    --batch-size 16 \
    --max-nodes 12000 \
    --steps 4 \
    --seed 20260924 \
    --motion-scale "${motion_scale}" \
    --initial-noise-scale "${initial_noise_scale}" \
    --disable-chi
}

run_eval apo_off 0.0 0.0
run_eval noisy_off 0.0 1.0
run_eval apo_learned 1.0 0.0
run_eval noisy_learned 1.0 1.0
