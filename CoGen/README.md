# CoGen

CoGen is a staged protein–ligand generation framework for conditional ligand
generation and protein-pocket conformational updating. This repository is a
clean code-and-checkpoint snapshot of the final development baseline. Large
training caches, trajectory files, and experiment run directories are
intentionally excluded from GitHub.

## What is included

- `ligand_generator/`: the 1,000-step ligand coordinate and atom-type
  diffusion model, its layerwise protein–ligand interaction extension, data
  preparation scripts, training scripts, sampling scripts, and tests;
- `pocketdiff/`: the residue-motion bridge model, TargetDiff-compatible
  27-dimensional protein atom features and 13-class ligand atom features,
  local-frame SE(3) updates, 20-step rollout, training scripts, and tests;
- `scripts/run_joint_inference.sh`: the staged joint-inference wrapper;
- `ligand_generator/pretrained_models/pretrained_diffusion.pt`: the base
  ligand-generator checkpoint;
- `ligand_generator/checkpoints/bridge_finetune_full/internal_adapter_best.pt`:
  the released ligand-generator adaptation branch;
- `pocketdiff/checkpoints/pocketdiff_v48_base.pt` and
  `pocketdiff/checkpoints/pocketdiff_noisy_ligand_ft.pt`: the released
  PocketDiff base and noisy-ligand-finetuned checkpoints;
- environment files, tests, upstream notices, and this README.

The following are deliberately omitted because they dominate repository size:
prepared `.pt` caches, MD trajectory data, bridge-state caches, training
`runs/`, duplicate intermediate checkpoints, and generated evaluation output.
Their expected paths and preparation commands are documented below.

## Model design

The ligand generator uses the TargetDiff-compatible 27-dimensional protein
atom representation and 13 ligand atom classes. Its retained UniTransformer-O2
backbone has one block, nine layers, 128 hidden channels, and 16 attention
heads. CoGen adds a time-conditioned asymmetric protein–ligand interaction
residual in the final three layers and a bounded, zero-initialized conservative
output adapter. The base backbone is frozen during the adaptation stages and
the added branch is optimized. The ligand generator predicts clean ligand
coordinates and atom classes while the protein coordinates remain fixed when
it is run alone.

PocketDiff uses the same atom-level input vocabulary and a residue-level
motion output. Protein atom features are pooled to residues, while the motion
heads also receive N–CA–C local frames, atom-resolved ligand contacts,
directional geometric messages, and peptide-sequence neighbors. The heads
predict conditional bridge velocity and apply local rigid SE(3) updates. The
released checkpoints use backbone-focused motion with chi updates disabled.
Each residue update preserves its internal rigid geometry; peptide continuity
is regularized by a soft loss rather than imposed as a hard constraint.

The joint schedule first runs ligand diffusion with the Apo protein fixed.
During the final 200 ligand reverse steps, PocketDiff is called every ten
steps, giving 20 protein updates. The updated protein coordinates are fed
back into subsequent ligand steps. This is a staged conditional bridge, not a
single end-to-end optimizer and not a reverse DDPM chain for the protein.

## Data contracts

The code expects the following prepared inputs, which are not distributed in
this repository:

| Input | Role | Preparation entry point |
|---|---|---|
| Apo2Mol prepared cache | PocketDiff training and validation | `pocketdiff/scripts/prepare_diffusion_cache.py` |
| Representative MD-frame cache | Ligand-generator dynamic adaptation | `ligand_generator/scripts/prepare_md_cache.py` |
| Bridge-state cache | Ligand-generator bridge fine-tuning | `ligand_generator/scripts/prepare_bridge_finetune_cache.py` |

Each selected MD frame is an independent ligand-generator sample. The full
development cache used 10 representative frames per source trajectory. The
public snapshot contains no structural dataset; users must provide data under
their own redistribution permissions.

## Environment

The validated development environment is the `targetdiff` conda environment.
The recorded runtime was Python 3.8.20, PyTorch 1.13.1, CUDA 11.6,
torch-geometric 2.2.0, torch-scatter 2.1.0, RDKit 2023.09.6, SciPy 1.10.1,
and NumPy 1.24.3. Create or adapt the environment from:

- `ligand_generator/environment.yaml`
- `pocketdiff/env/targetdiff_environment.yaml`
- `pocketdiff/env/runtime.txt`

From the repository root:

```bash
conda activate targetdiff
export PYTHONPATH="$PWD/pocketdiff:$PWD/ligand_generator"
export POCKETDIFF_TARGETDIFF_ROOT="$PWD/ligand_generator"
```

## Training

Training requires external prepared caches. The commands below write new
outputs and do not overwrite the packaged checkpoints.

PocketDiff base training on four GPUs:

```bash
cd pocketdiff
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  PREPARED_CACHE="$PWD/data/prepared_diffusion_full_v47.pt" \
  OUTPUT_DIR="$PWD/runs/retrain_v48" \
  bash run_v48_full_ddp.sh
```

PocketDiff noisy-ligand fine-tuning:

```bash
cd pocketdiff
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  INIT_CHECKPOINT="$PWD/checkpoints/pocketdiff_v48_base.pt" \
  PREPARED_CACHE="$PWD/data/prepared_diffusion_full_v47.pt" \
  OUTPUT_DIR="$PWD/runs/retrain_noisy_ligand" \
  bash run_noisy_ligand_finetune_3000_ddp.sh
```

Ligand-generator dynamic adaptation and bridge fine-tuning use the scripts in
`ligand_generator/scripts/`. Set their cache paths explicitly because the
large caches are not part of this snapshot.

## Joint inference

Joint inference needs one prepared PocketDiff cache containing the input
complexes. Set `COGEN_POCKET_CACHE` to that file before running:

```bash
COGEN_POCKET_CACHE=/path/to/prepared_diffusion_cache.pt \
COGEN_SAMPLES=1 COGEN_DEVICE=cuda:0 \
COGEN_OUTPUT_DIR="$PWD/results/joint_smoke" \
  bash scripts/run_joint_inference.sh
```

The wrapper runs 1,000 ligand reverse steps and performs 20 PocketDiff calls
in the final 200 ligand steps. It writes the staged trajectory and comparison
records to `COGEN_OUTPUT_DIR`. The input ligand atom count follows the prepared
reference complex; this release therefore represents a known-atom-count bridge
test rather than an unrestricted ligand-size benchmark.

The wrapper defaults to the packaged checkpoints. Override them with
`COGEN_POCKET_CHECKPOINT`, `COGEN_POCKET_ADAPTER_CHECKPOINT`,
`COGEN_ORIGINAL_CHECKPOINT`, and `COGEN_INTERNAL_CHECKPOINT` when testing
another checkpoint pair.

## Validation

The release inventory and Python syntax can be checked without any dataset or
GPU:

```bash
python scripts/validate_release.py
```

The model-level validators and unit tests require the external caches and the
runtime dependencies. They are intentionally kept as source code so the same
checks can be run after the data paths are supplied.

## Scope and evidence

This repository records the final engineering baseline and its reproducible
training/inference entry points. It does not claim that the packaged model
outperforms every baseline. The release manifest records the training and
evaluation provenance available at packaging time. Upstream licenses,
checkpoint terms, and structural-data redistribution permissions must be
reviewed before redistribution.
