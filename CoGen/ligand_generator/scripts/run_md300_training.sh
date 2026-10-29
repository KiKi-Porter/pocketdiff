#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/targetdiff/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/md_dynamic.yml}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained_models/pretrained_diffusion.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/md_dynamic_300}"
DEVICE="${DEVICE:-cuda}"

cd "${ROOT}"
PYTHONPATH="${ROOT}" "${PYTHON}" scripts/train_md_dynamic.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
