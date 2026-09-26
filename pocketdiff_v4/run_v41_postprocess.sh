#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="pocketdiff_v4/runs/v41_3000_s4_20260924"
SUMMARY="${RUN_DIR}/train_summary.json"
EVAL_DIR="${RUN_DIR}/controls_best"

while [[ ! -f "${SUMMARY}" ]]; do
  if ! pgrep -f "pocketdiff_v4.train.*${RUN_DIR}" >/dev/null 2>&1; then
    echo "training process disappeared before completion" >&2
    exit 2
  fi
  sleep 30
done

STATUS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "${SUMMARY}")"
if [[ "${STATUS}" != "completed" ]]; then
  echo "training summary status is ${STATUS}, refusing evaluation" >&2
  exit 3
fi

CHECKPOINT="${RUN_DIR}/best.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${RUN_DIR}/latest.pt"
fi

exec env CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
  bash pocketdiff_v4/run_v41_evaluate_controls.sh "${CHECKPOINT}" "${EVAL_DIR}"
