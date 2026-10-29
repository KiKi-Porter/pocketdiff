#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/modelcoding/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# The container exposes a 64 MiB /dev/shm.  NCCL's shared-memory transport
# fails during communicator setup even after stale segments are removed, so
# use the socket transport explicitly for this four-GPU run.
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/pocketdiff/runs/v46_final_3000_ddp}"
PREPARED_CACHE="${PREPARED_CACHE:-${ROOT}/pocketdiff/data/prepared_diffusion_v46_3000.pt}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  pocketdiff/scripts/train_diffusion_ddp.py \
  --output-dir "${OUTPUT_DIR}" \
  --prepared-cache "${PREPARED_CACHE}" \
  --device cuda \
  --train-count 3000 \
  --valid-count 300 \
  --test-count 300 \
  --holdout-count 300 \
  --updates 2000 \
  --batch-size 16 \
  --checkpoint-every 100 \
  --gradient-accumulation-steps 1 \
  --encoder-backend targetdiff \
  --architecture-variant v46 \
  --hidden-dim 128 \
  --prediction-type bridge_velocity \
  --target-parameterization bridge_velocity \
  --target-steps 20 \
  --sampler-steps 20,50,100 \
  --backbone-only \
  --disable-chi \
  --endpoint-weight 0.0 \
  --backbone-endpoint-weight 0.25 \
  --next-state-weight 1.0 \
  --next-state-steps 20 \
  --chi-loss-weight 0.0 \
  --continuity-weight 0.10 \
  --direction-weight 0.0 \
  --translation-noise-scale 0.0 \
  --rotation-noise-scale 0.0 \
  --chi-noise-scale 0.0 \
  --self-state-probability 0.25 \
  --self-state-steps 4 \
  --self-state-rollout-steps 20 \
  --self-state-warmup 400 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0

CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
"${PYTHON}" pocketdiff/scripts/evaluate_diffusion.py \
  --checkpoint "${OUTPUT_DIR}/checkpoint.pt" \
  --prepared-cache "${PREPARED_CACHE}" \
  --device cuda:0 \
  --valid-count 300 \
  --test-count 300 \
  --steps 20,50,100 \
  --output "${OUTPUT_DIR}/evaluation_20_50_100.json"
