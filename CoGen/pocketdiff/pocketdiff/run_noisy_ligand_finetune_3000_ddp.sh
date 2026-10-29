#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
TARGETDIFF_ROOT="${TARGETDIFF_ROOT:-$WORKSPACE/targetdiff-main/targetdiff-main}"
TARGETDIFF_CHECKPOINT="${TARGETDIFF_CHECKPOINT:-$TARGETDIFF_ROOT/pretrained_models/pretrained_diffusion.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/noisy_ligand_ft_3000_20261021}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/init_checkpoint_best.pt}"
TRAIN_COUNT="${TRAIN_COUNT:-3000}"
VALID_COUNT="${VALID_COUNT:-300}"
TEST_COUNT="${TEST_COUNT:-300}"
HOLDOUT_COUNT="${HOLDOUT_COUNT:-$VALID_COUNT}"
UPDATES="${UPDATES:-470}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-47}"
VALIDATE_EVERY="${VALIDATE_EVERY:-47}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export POCKETDIFF_TARGETDIFF_ROOT="$TARGETDIFF_ROOT"
export NCCL_SHM_DISABLE=1

exec conda run --no-capture-output -n targetdiff torchrun \
  --standalone \
  --nproc_per_node=4 \
  pocketdiff/scripts/train_diffusion_ddp.py \
  --prepared-cache "$ROOT/data/prepared_diffusion_full_v47.pt" \
  --output-dir "$OUTPUT_DIR" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --targetdiff-checkpoint "$TARGETDIFF_CHECKPOINT" \
  --targetdiff-root "$TARGETDIFF_ROOT" \
  --ligand-noise-t-min 0 \
  --ligand-noise-t-max 199 \
  --ligand-clean-probability 0.20 \
  --train-count "$TRAIN_COUNT" \
  --valid-count "$VALID_COUNT" \
  --test-count "$TEST_COUNT" \
  --holdout-count "$HOLDOUT_COUNT" \
  --train-seed 5700 \
  --holdout-seed 5701 \
  --diffusion-seed 5700 \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps 1 \
  --self-state-probability 0.25 \
  --self-state-steps 4 \
  --self-state-rollout-steps 20 \
  --self-state-warmup 15 \
  --apo-boundary-probability 0.25 \
  --times 0.05,0.25,0.5,0.75,0.95,1.0 \
  --time-min 0.02 \
  --time-max 1.0 \
  --learning-rate 2e-5 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --hidden-dim 128 \
  --dropout 0.0 \
  --encoder-backend targetdiff \
  --architecture-variant v47 \
  --endpoint-weight 0.0 \
  --next-state-weight 5.0 \
  --next-state-steps 20 \
  --chi-loss-weight 0.0 \
  --backbone-only \
  --disable-chi \
  --motion-parameterization remaining \
  --sampler-steps 20,50,100 \
  --translation-noise-scale 0.0 \
  --rotation-noise-scale 0.0 \
  --chi-noise-scale 0.0 \
  --prediction-type bridge_velocity \
  --target-parameterization bridge_velocity \
  --target-steps 20 \
  --backbone-endpoint-weight 2.0 \
  --continuity-weight 0.2 \
  --direction-weight 0.0 \
  --backbone-progress-direction-weight 0.1 \
  --backbone-progress-magnitude-weight 0.25 \
  --backbone-progress-threshold 0.05 \
  --motion-bucket-count 4 \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --validate-every "$VALIDATE_EVERY"
