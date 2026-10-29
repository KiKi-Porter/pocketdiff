#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/modelcoding/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/Apo2Mol-main/Apo2MOl-dataset/data_folder}"
SPLIT_PATH="${SPLIT_PATH:-${ROOT}/Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl}"
CACHE="${CACHE:-${ROOT}/pocketdiff/data/prepared_diffusion_full_v47.pt}"
RUN_DIR="${RUN_DIR:-${ROOT}/pocketdiff/runs/v47_full_dataset_20260930}"
PREP_LOG="${PREP_LOG:-${RUN_DIR}/prepare.log}"
TRAIN_LOG="${TRAIN_LOG:-${RUN_DIR}/train.log}"
STATUS_FILE="${STATUS_FILE:-${RUN_DIR}/supervisor_status.json}"
START_TRAINING="${START_TRAINING:-0}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export POCKETDIFF_TORCH_THREADS="${POCKETDIFF_TORCH_THREADS:-1}"
export POCKETDIFF_TORCH_INTEROP_THREADS="${POCKETDIFF_TORCH_INTEROP_THREADS:-1}"

mkdir -p "${RUN_DIR}"

write_status() {
  local status="$1"
  "${PYTHON}" -c 'import json, os, sys; path, status = sys.argv[1:]; json.dump({"status": status, "pid": os.getpid()}, open(path, "w"), indent=2); open(path, "a").write("\n")' \
    "${STATUS_FILE}" "${status}"
}

write_status "preparing_cache"

if [[ ! -f "${CACHE}" || ! -f "${CACHE%.pt}.manifest.json" ]]; then
  "${PYTHON}" "${ROOT}/pocketdiff/scripts/prepare_diffusion_cache.py" \
    --data-root "${DATA_ROOT}" \
    --split-path "${SPLIT_PATH}" \
    --output "${CACHE}" \
    --fixed-steps 20 \
    --progress-every 250 \
    >"${PREP_LOG}" 2>&1
fi

read -r TRAIN_COUNT VALID_COUNT TEST_COUNT < <(
  "${PYTHON}" -c '
import json, sys
from pathlib import Path
from pocketdiff.preprocessing.diffusion_cache import load_diffusion_cache
cache = Path(sys.argv[1])
payload = load_diffusion_cache(cache)
counts = {key: len(payload["splits"][key]) for key in ("train", "valid", "test")}
manifest = json.loads(cache.with_suffix(".manifest.json").read_text())
if manifest.get("cache_hash") != payload.get("cache_hash"):
    raise SystemExit("manifest/cache hash mismatch")
print(counts["train"], counts["valid"], counts["test"])
' "${CACHE}"
)

write_status "cache_ready_waiting_for_manual_training"

if [[ "${START_TRAINING}" != "1" ]]; then
  exit 0
fi

write_status "training_starting"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" \
OUTPUT_DIR="${RUN_DIR}" \
PREPARED_CACHE="${CACHE}" \
TRAIN_COUNT="${TRAIN_COUNT}" \
VALID_COUNT="${VALID_COUNT}" \
TEST_COUNT="${TEST_COUNT}" \
HOLDOUT_COUNT="${VALID_COUNT}" \
  "${ROOT}/pocketdiff/run_v47_full_ddp.sh" >"${TRAIN_LOG}" 2>&1

write_status "training_finished"
