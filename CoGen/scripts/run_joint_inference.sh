#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR_ROOT="$RELEASE_ROOT/ligand_generator"
OUTPUT_DIR="${COGEN_OUTPUT_DIR:-$RELEASE_ROOT/results/joint_inference}"
DEVICE="${COGEN_DEVICE:-cuda}"
SAMPLES="${COGEN_SAMPLES:-10}"
SEED="${COGEN_SEED:-20261028}"
POCKET_CACHE="${COGEN_POCKET_CACHE:-}"
POCKET_CHECKPOINT="${COGEN_POCKET_CHECKPOINT:-$RELEASE_ROOT/pocketdiff/checkpoints/pocketdiff_v48_base.pt}"
POCKET_ADAPTER_CHECKPOINT="${COGEN_POCKET_ADAPTER_CHECKPOINT:-$RELEASE_ROOT/pocketdiff/checkpoints/pocketdiff_noisy_ligand_ft.pt}"
ORIGINAL_CHECKPOINT="${COGEN_ORIGINAL_CHECKPOINT:-$GENERATOR_ROOT/pretrained_models/pretrained_diffusion.pt}"
INTERNAL_CHECKPOINT="${COGEN_INTERNAL_CHECKPOINT:-$GENERATOR_ROOT/checkpoints/bridge_finetune_full/internal_adapter_best.pt}"

if [[ -z "$POCKET_CACHE" ]]; then
  echo "COGEN_POCKET_CACHE must point to an external prepared PocketDiff cache" >&2
  exit 2
fi

cd "$GENERATOR_ROOT"
export PYTHONPATH="$GENERATOR_ROOT:$RELEASE_ROOT/pocketdiff${PYTHONPATH:+:$PYTHONPATH}"
export POCKETDIFF_TARGETDIFF_ROOT="$GENERATOR_ROOT"

python scripts/bridge_targetdiff_pocketdiff.py \
  --pocket-cache "$POCKET_CACHE" \
  --pocket-checkpoint "$POCKET_CHECKPOINT" \
  --pocket-adapter-checkpoint "$POCKET_ADAPTER_CHECKPOINT" \
  --original-checkpoint "$ORIGINAL_CHECKPOINT" \
  --internal-checkpoint "$INTERNAL_CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --samples "$SAMPLES" \
  --seed "$SEED" \
  --full
