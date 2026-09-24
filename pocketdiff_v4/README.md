# PocketDiff v4

PocketDiff v4 is an independent residue-level, DynamicBind-inspired model
candidate. It does not alter the frozen v3 implementation or checkpoint.

## Architecture

- Protein atoms are grouped into residue nodes with categorical residue type,
  pooled atom features, and an N-CA-C local frame rebuilt from the current
  protein coordinates.
- A cached apo residue-radius graph carries residue-residue messages. A second
  cached graph connects ligand atoms to residue nodes; ligand atom identity,
  position, and local molecular geometry remain atom-resolved.
- Message blocks use invariant scalar channels and covariant vector channels.
  Scalar gates act on relative-coordinate directions and transported vectors.
  The residue heads produce local translation, local SO(3) rotation, and five
  periodic side-chain chi increments.
- The coordinate solver applies a residue-local rigid transform and then
  topology-defined sparse chi rotations. Its interface is shared by training
  rollouts and inference.
- The training objective is intended to include randomized bridge states,
  detached self-generated intermediate states, endpoint coordinate loss, and
  bounded per-step motion. Training and sampler implementation must preserve
  this state/update contract.
- At every rollout state, the motion target is recomputed from the exact
  current-to-holo local-frame bridge, divided by remaining steps, then
  projected into the translation/rotation/chi ranges the heads can emit. The
  same rigid-plus-chi solver applies model predictions before a direct
  coordinate endpoint loss is measured. Motion and endpoint losses are
  normalized per complex before batch averaging, avoiding a node-count bias.

The custom scalar/vector implementation avoids an `e3nn` runtime dependency.
Equivariance is enforced by construction for coordinate updates: local scalar
predictions are invariant, local frames rotate with the input, and global
vectors are reconstructed from those frames.

## Cache and Label Boundary

Run on CPU with the project environment:

```bash
PYTHONPATH=. conda run -n targetdiff python -m pocketdiff_v4.cache \
  --source pocketdiff_v1/data/apo2mol_3000_cache.pt \
  --output pocketdiff_v4/data/residue_graphs.pt
```

The versioned cache stores model-side features under each record's `input`
key, while holo endpoint coordinates and chi supervision labels are isolated
under `target`. Inference and reverse-prior initialization may consume only
`input`; `target` is used to construct training loss and, after training, to
score predictions. Holo protein coordinates or holo displacement must never
initialize inference state.

The residue chi geometry mask is derived only from atom topology. The separate
chi supervision mask intersects that topology with labels observed in both
apo/holo structures and is not a solver or model input.

## Validation Gates

1. Run `python -m pytest pocketdiff_v4/tests`.
2. Build and reload the fixed 3000/300/300 CPU cache; verify its source
   fingerprint, exact sample identities, and disjoint split IDs.
3. Add model-level equivariance and finite forward/backward tests.
4. Pass bounded single-complex overfit and multi-step rollout stability tests.
5. Train on all 3000 training examples and reload the final checkpoint.
6. Only after training completes, evaluate train/valid/test with the frozen v3
   split and compare paired sample predictions. Do not replace v3 unless v4 is
   no worse on the predeclared validation/test criteria.

Stability smoke tests may inspect losses and finite coordinates; they are not
apo/holo performance evaluations. No final performance evaluation is allowed
before full training has completed.

## Medium Run

The validated CPU cache is `pocketdiff_v4/data/residue_graphs_medium.pt`
(3000/300/300 records, source split IDs preserved). To launch the detached
medium run and its post-training evaluations:

```bash
bash pocketdiff_v4/run_medium_and_evaluate.sh
```

The runner uses physical GPUs 4–7 with four Gloo ranks, per-rank batch size 64,
50 epochs, five-step training rollouts, and zero DataLoader workers. It saves
`latest.pt` every epoch. After the 600-update training summary confirms
completion, it evaluates all three splits for v4, evaluates the frozen v3
checkpoint with four Gaussian-initialized steps (no VP-prior label leakage),
and writes a paired comparison. Artifacts are isolated under
`pocketdiff_v4/runs/medium_3000_20260923/`.

## Run Result

The 600-update run completed in 208.7 seconds with finite losses and
checkpoints. The raw four-step sampler (initial-noise scale 1.0) regressed:
all-protein RMSD was 0.5229 Å on valid and 0.5145 Å on test, versus apo at
0.4520/0.4456 Å and v3 Gaussian at 0.4679/0.4674 Å.

Validation-only calibration found that setting initial-noise scale to 0.0
nearly reproduces apo. Its final all-protein RMSD was 0.4518 Å valid and
0.4459 Å test; mean coordinate displacement was only 0.008 Å. It beats v3's
noisy Gaussian sampler but does not demonstrate useful learned motion and is
0.00032 Å worse than apo on test. Keep v3 as the accepted model; do not treat
this no-op-like output as a successful v4 baseline.
