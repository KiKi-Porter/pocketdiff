# TargetDiff Internal Cross Final

This directory is a self-contained snapshot of the final TargetDiff-based
model used in the PocketDiff bridge project.

Release date: 2026-10-03

The release keeps the original TargetDiff data representation and diffusion
protocol, while adding a conservative, time-conditioned, layerwise
protein-ligand interaction branch inside the UniTransformer backbone.

## 1. Scope

The model is a TargetDiff ligand-generation model, not a PocketDiff protein
motion model. Protein coordinates remain fixed during TargetDiff sampling.
This snapshot is intended to be the stable TargetDiff side of the later
PocketDiff bridge experiment.

The release contains:

- complete Python source code;
- training, evaluation, sampling, and validation scripts;
- model and training configurations;
- original TargetDiff pretrained weights;
- the 3000-sample development checkpoint;
- the full dynamic-data checkpoint;
- prepared TargetDiff `.pt` caches;
- test code and recorded evaluation results.

The large `core` file from the development workspace is a crash dump and is
intentionally excluded.

## 2. Data Representation Contract

The following representation is unchanged from TargetDiff:

- protein atom features: 27 dimensions;
- ligand atom classes: 13 dimensions/classes;
- protein coordinates are fixed during TargetDiff sampling;
- ligand coordinates use the original Gaussian diffusion;
- ligand atom types use the original categorical diffusion;
- 1000 diffusion timesteps;
- `C0` coordinate prediction;
- `UniTransformerO2TwoUpdateGeneral` backbone;
- original diffusion objective:

```text
loss = loss_pos + 100 * loss_v
```

The full prepared cache is:

```text
data/full_dynamic/prepared_targetdiff_full.pt
```

It contains 22,145 valid frame samples:

```text
train: 17779
valid:  2070
test:   2296
```

The source data comes from 2,347 MD trajectories. Ten cluster representative
frames were selected per trajectory, and every representative frame is an
independent TargetDiff training sample. The cache is ready for direct loading;
training does not rebuild the MD topology or pocket graphs.

## 3. Architecture

### 3.1 Original TargetDiff path

The original path is retained:

```text
protein atom feature (27D)
ligand atom class (13 classes)
diffusion timestep
        |
        v
protein/ligand hidden embeddings
        |
        v
UniTransformerO2TwoUpdateGeneral
        |
        +--> ligand coordinate prediction
        +--> ligand atom-type prediction
```

The original attention, edge construction, radial features, coordinate
updates, categorical diffusion, and sampling equations are preserved.

### 3.2 Conservative Adapter

`models/conservative_adapter.py` adds a small residual protein-ligand
interaction adapter after the main refinement path.

Properties:

- uses ligand-protein relative distances and directions;
- coordinate correction is built from relative unit vectors and is rotation
  equivariant;
- correction magnitude is bounded;
- atom-type correction is a small residual logit correction;
- the new branch is zero initialized;
- baseline anchor losses keep the adapted model close to the original model.

### 3.3 Internal Cross Interaction

`models/layerwise_cross.py` implements
`LayerwiseCrossInteraction`.

It is inserted by `models/uni_transformer.py` after the final three
Transformer layers. It uses:

- current protein and ligand hidden states;
- asymmetric protein-ligand cross edges;
- TargetDiff edge attributes;
- distance Gaussian features;
- diffusion timestep embedding;
- relative coordinate directions.

It returns:

- a scalar hidden-state residual;
- a ligand-only coordinate residual.

The coordinate branch does not update protein coordinates. Both residual gates
are zero initialized and the coordinate correction is bounded by
`internal_cross_max_coord_delta`.

The effective modification is therefore:

```text
original UniTransformer
        +
time-conditioned layerwise asymmetric cross residuals
        +
conservative output adapter
```

The original model is not replaced, and the original TargetDiff data
language is not changed.

## 4. Important Checkpoints

Original TargetDiff:

```text
pretrained_models/pretrained_diffusion.pt
```

3000-sample development checkpoint:

```text
checkpoints/internal_cross_md300_20261003/internal_adapter_best.pt
```

Full dynamic-data checkpoint:

```text
checkpoints/full_dynamic_internal_cross_20261003/internal_adapter_best.pt
```

The full-data checkpoint is the release checkpoint.

## 5. Environment

The expected environment is the existing `targetdiff` conda environment.
The environment specification is included in:

```text
environment.yaml
```

Example:

```bash
conda env update -n targetdiff -f environment.yaml
conda activate targetdiff
```

Run commands from this directory with:

```bash
export PYTHONPATH=.
```

## 6. Training

### Full-data training configuration

```text
configs/full_dynamic_internal_cross.yml
```

The full cache is already prepared. No preprocessing is required for
retraining with the included cache.

The released full-data run used:

- 22,145 cached samples;
- batch size 2 per GPU;
- two-GPU DDP;
- GPUs 2 and 3;
- three complete epochs;
- 13,335 optimizer steps;
- frozen original Transformer;
- trainable Conservative Adapter and Internal Cross branch;
- `num_workers=0` because the host has a 64 MB `/dev/shm`;
- original 1000-step diffusion schedule.

Example two-GPU command:

```bash
NCCL_SHM_DISABLE=1 \
CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH=. \
torchrun --standalone --nproc_per_node=2 \
  scripts/train_md_dynamic.py \
  --config configs/full_dynamic_internal_cross.yml \
  --checkpoint checkpoints/internal_cross_md300_20261003/internal_adapter_best.pt \
  --output-dir runs/full_dynamic_internal_cross_retrain \
  --device cuda \
  --only-stage internal_adapter \
  --epochs 3 \
  --num-workers 0
```

The training script saves:

- `internal_adapter_best.pt`;
- `best.pt`;
- `internal_adapter_history.jsonl`;
- `training_summary.json`.

Validation checkpoint selection uses the diffusion validation objective
(diffusion loss plus adapter anchor loss), not generated-molecule RMSD.

## 7. Diffusion Sampling

Sampling uses the original TargetDiff reverse diffusion process. It is a
1000-step categorical-plus-Gaussian diffusion model, not the PocketDiff
flow-matching model.

The main sampling entry points are in:

```text
models/molopt_score_model.py
scripts/compare_generation_50.py
```

For a paired generation comparison:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python scripts/compare_generation_50.py \
  --old-checkpoint pretrained_models/pretrained_diffusion.pt \
  --new-checkpoint checkpoints/full_dynamic_internal_cross_20261003/internal_adapter_best.pt \
  --cache data/full_dynamic/prepared_targetdiff_full.pt \
  --output-dir results/generation_compare_retrain \
  --num-samples 50 \
  --steps 1000 \
  --seed 20261002 \
  --device cuda:0
```

The comparison uses:

- the same held-out pockets;
- the same ligand atom-count prior;
- the same initial ligand coordinates and atom classes;
- the same reverse-diffusion random stream;
- no reference ligand coordinates or atom types as model input.

## 8. Evaluation

### Diffusion loss

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_md_dynamic.py \
  --config configs/full_dynamic_internal_cross.yml \
  --cache data/full_dynamic/prepared_targetdiff_full.pt \
  --split test \
  --checkpoint pretrained_models/pretrained_diffusion.pt \
  --checkpoint checkpoints/internal_cross_md300_20261003/internal_adapter_best.pt \
  --checkpoint checkpoints/full_dynamic_internal_cross_20261003/internal_adapter_best.pt \
  --device cuda \
  --batch-size 8 \
  --num-workers 0 \
  --output results/test_loss_comparison_reproduced.json
```

### Molecule quality

The generation comparison reports:

- sampling completion;
- distance-based molecule stability;
- reconstruction success;
- connected reconstruction rate;
- QED;
- synthetic accessibility;
- LogP;
- Lipinski score;
- severe protein-ligand clash count;
- nearest protein distance;
- bond-length distribution divergence;
- fingerprint diversity;
- protein-ligand distance distribution divergence.

These are screening diagnostics. They do not establish binding affinity or
experimental activity.

## 9. Recorded Results

The full-data test diffusion loss comparison is stored in:

```text
results/eval_full_dynamic_internal_cross_20261003/test_loss_comparison.json
```

Recorded test objective:

| Model | Test objective |
|---|---:|
| Original TargetDiff | 3.93279 |
| 3000-sample Internal Cross | 3.85341 |
| Full-data Internal Cross | 3.61170 |

The 50-pocket full-data generation comparison is stored in:

```text
results/generation_compare_full_dynamic50_20261003/report.md
results/generation_compare_full_dynamic50_20261003/summary.json
```

On that fixed 50-pocket, 1000-step comparison:

| Metric | Original TargetDiff | Full-data Internal Cross |
|---|---:|---:|
| Sampling completion | 100% | 100% |
| Molecule stability | 28% | 14% |
| Reconstruction success | 100% | 90% |
| Connected reconstruction | 68% | 60% |
| QED | 0.256 | 0.273 |
| SA | 0.544 | 0.519 |

The lower diffusion loss does not automatically imply better long-chain
generation. The included full-data checkpoint is therefore the final
research snapshot, but the original TargetDiff checkpoint remains the safety
baseline for generation-quality regression checks.

## 10. Verification

Run syntax compilation:

```bash
PYTHONPATH=. python -m py_compile \
  models/*.py \
  datasets/*.py \
  scripts/*.py \
  utils/*.py
```

Run the focused architecture tests:

```bash
PYTHONPATH=. pytest -q \
  tests/test_aplc.py \
  tests/test_conservative_adapter.py \
  tests/test_layerwise_cross.py
```

Load the included full cache:

```bash
PYTHONPATH=. python - <<'PY'
import torch

payload = torch.load(
    "data/full_dynamic/prepared_targetdiff_full.pt",
    map_location="cpu",
    weights_only=False,
)
assert len(payload["samples"]) == 22145
assert payload["metadata"]["feature_contract"]["protein_atom_feature_dim"] == 27
assert payload["metadata"]["feature_contract"]["ligand_atom_class_dim"] == 13
print("cache verification passed")
PY
```

Load the release checkpoint:

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
import torch

checkpoint = Path(
    "checkpoints/full_dynamic_internal_cross_20261003/internal_adapter_best.pt"
)
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert "model" in payload
assert payload["stage"] == "internal_adapter"
assert payload["iteration"] == 13335
print("checkpoint verification passed")
PY
```

## 11. Directory Layout

```text
targetdiff_internal_cross_final_20261003/
├── configs/
├── core/
├── datasets/
├── examples/
├── models/
├── scripts/
├── tests/
├── utils/
├── pretrained_models/
│   └── pretrained_diffusion.pt
├── data/
│   ├── md300/
│   └── full_dynamic/
│       └── prepared_targetdiff_full.pt
├── checkpoints/
│   ├── internal_cross_md300_20261003/
│   └── full_dynamic_internal_cross_20261003/
├── results/
│   ├── eval_full_dynamic_internal_cross_20261003/
│   └── generation_compare_full_dynamic50_20261003/
├── environment.yaml
└── README.md
```

