#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/pocketdiff_v4/runs/medium_3000_20260923"
mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
cd "$ROOT"

finish() {
  result=$?
  printf '%s\n' "$result" > "$RUN_DIR/job_exit_code"
}
trap finish EXIT

printf 'training_started %s\n' "$(date -u +%FT%TZ)" >> "$RUN_DIR/job_state.log"
env PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
  conda run -n targetdiff torchrun --standalone --nproc_per_node=4 \
  -m pocketdiff_v4.train \
  --data pocketdiff_v4/data/residue_graphs_medium.pt \
  --output-dir pocketdiff_v4/runs/medium_3000_20260923 \
  --epochs 50 \
  --batch-size 64 \
  --max-steps 5 \
  --num-workers 0 \
  --save-every 12 \
  --log-every 12 \
  --lr 2e-4 \
  > "$RUN_DIR/train.log" 2>&1

python -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); assert d["status"] == "completed" and d["updates"] == 600, d' \
  "$RUN_DIR/train_summary.json"
printf 'training_completed %s\n' "$(date -u +%FT%TZ)" >> "$RUN_DIR/job_state.log"

env PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
  conda run -n targetdiff torchrun --standalone --nproc_per_node=4 \
  -m pocketdiff_v4.evaluate \
  --data pocketdiff_v4/data/residue_graphs_medium.pt \
  --checkpoint pocketdiff_v4/runs/medium_3000_20260923/latest.pt \
  --output pocketdiff_v4/runs/medium_3000_20260923/evaluation_v4.json \
  --batch-size 32 \
  --max-nodes 16000 \
  --steps 4 \
  --seed 20260923 \
  > "$RUN_DIR/evaluate_v4.log" 2>&1
printf 'v4_evaluation_completed %s\n' "$(date -u +%FT%TZ)" >> "$RUN_DIR/job_state.log"

env PYTHONPATH="$ROOT/pocketdiff_v3:$ROOT" CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
  conda run -n targetdiff torchrun --standalone --nproc_per_node=4 \
  -m evaluate \
  --data pocketdiff_v3/data/prepared_dataset.pt \
  --checkpoint pocketdiff_v3/runs/v3_3000/checkpoint.pt \
  --output pocketdiff_v4/runs/medium_3000_20260923/evaluation_v3_gaussian.json \
  --batch-size 40 \
  --max-nodes 10500 \
  --steps 4 \
  --seed 20260923 \
  --initial-state gaussian \
  --seed-mode sample \
  > "$RUN_DIR/evaluate_v3.log" 2>&1
printf 'v3_evaluation_completed %s\n' "$(date -u +%FT%TZ)" >> "$RUN_DIR/job_state.log"

env PYTHONPATH="$ROOT" conda run -n targetdiff python -m pocketdiff_v4.compare \
  --v4 pocketdiff_v4/runs/medium_3000_20260923/evaluation_v4.json \
  --v3 pocketdiff_v4/runs/medium_3000_20260923/evaluation_v3_gaussian.json \
  --output pocketdiff_v4/runs/medium_3000_20260923/paired_comparison.json \
  > "$RUN_DIR/compare.log" 2>&1
printf 'comparison_completed %s\n' "$(date -u +%FT%TZ)" >> "$RUN_DIR/job_state.log"
