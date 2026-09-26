# PocketDiff v4.4.0

PocketDiff v4 is an independent residue-level, DynamicBind-inspired model
candidate. It does not alter the frozen v3 implementation or checkpoint.

The v4.4 objective is deliberately backbone-first: given apo protein
coordinates and ligand atom conditions, training emphasizes the N/CA/C/O
backbone transition toward holo. Side-chain chi and all-atom accuracy remain
secondary until the backbone field is demonstrably useful.

## Architecture

- Protein atoms are grouped into residue nodes with categorical residue type,
  pooled atom features, and an N-CA-C local frame rebuilt from the current
  protein coordinates.
- A cached apo residue-radius graph carries residue-residue messages. A second
  cached graph connects ligand atoms to residue nodes; ligand atom identity,
  position, and local molecular geometry remain atom-resolved.
- Message blocks use invariant scalar channels and covariant vector channels.
  Scalar gates act on relative-coordinate directions and transported vectors.
  The residue heads predict complete remaining local translation, local SO(3)
  rotation, and four periodic side-chain chi increments.
- The coordinate solver applies a residue-local rigid transform and then
  topology-defined sparse chi rotations. Its interface is shared by training
  rollouts and inference.
- The model predicts a complete current-to-holo remaining transform. The
  sampler chooses the update fraction without changing the model target.
- v4.3.3 adds residue-balanced CA displacement and CA direction losses on top
  of the v4.3.2 complete-remaining-transform training contract. Initial weights
  are 0.1 for CA displacement and 0.05 for CA direction; both are logged
  separately. Inference motion scale remains outside the training loss.
- v4.3.2 keeps the deployment sampler calibration while training the model on
  the complete remaining transform. Training logs report rigid, bridge,
  endpoint, and direction loss components.
- v4.3.1 uses fixed four-step autonomous training rollouts and a validation-
  selected fixed `0.20` relaxation fraction. Checkpoint selection uses
  autonomous validation backbone RMSD.
- Rigid-only endpoint supervision is restricted to backbone atoms. The final
  endpoint term is added outside the per-step loss average. Direction loss is
  disabled by default and uses a `0.05 Å` target-motion threshold when enabled.
- v4.4 adds `--backbone-only-objective`, masking bridge, final endpoint, and
  truncated endpoint coordinate losses to the cached backbone mask. Checkpoint
  selection remains autonomous validation backbone RMSD.

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
4. Compare sampler schedules on validation only, then freeze the selected
   schedule before test evaluation.
5. Pass bounded single-complex overfit and autonomous multi-step stability
   tests.
6. Train on all 3000 training examples and reload the selected checkpoint.
7. Evaluate train/valid/test with the frozen split and paired bootstrap
   intervals for CA, backbone, and all-atom RMSD. Keep v3 as the accepted
   baseline unless v4 is no worse on predeclared held-out criteria.

Stability smoke tests may inspect losses and finite coordinates; they are not
apo/holo performance evaluations. No final performance evaluation is allowed
before full training has completed.

## v4.3.1 Run

The validated CPU cache is `pocketdiff_v4/data/residue_graphs_v42_contract.pt`
(3000/300/300 records, source split IDs preserved). Launch training with:

```bash
bash pocketdiff_v4/run_v431_3000.sh
```

The runner uses GPUs 4–7 with four Gloo ranks, per-rank batch size 64, fixed
four-step autonomous rollouts, zero initial protein noise, no direction loss,
and the validation-selected fixed `0.20` schedule. Evaluation reads the frozen
schedule from checkpoint metadata unless explicitly overridden. Checkpoints
store source-cache and prepared-cache fingerprints plus the model geometry
configuration.

Before v4.3.1 training, the existing v4.3 checkpoint was compared on all 300
validation complexes. `fixed_020` achieved mean backbone RMSD improvement of
`0.01331 Å` and all-atom improvement of `0.01107 Å`; the old remaining-fraction
schedule regressed backbone RMSD by `0.01110 Å`. These are validation-only
sampler-calibration results, not a new model result.

## v4.4 Backbone-Focused Run

The v4.4 runner uses GPUs 4-7, 2400 updates, two-step autonomous rollouts,
zero initial protein noise, no chi updates, zero CA direction-loss weight, and
backbone-only bridge/endpoint supervision:

```bash
bash pocketdiff_v4/run_v44_3000.sh
```

The completed run is stored in
`pocketdiff_v4/runs/v440_launch_probe_20260925`. Its clean test result was
`+0.00622 Å` CA RMSD improvement and `+0.00408 Å` backbone RMSD improvement;
the paired 95% backbone interval was `[-0.00349, 0.01277] Å`, so this is
non-regressive but not a statistically confirmed upgrade over v4.3.4.
Direction cosine was `0.07695`, indicating that ligand-conditioned backbone
direction learning remains the primary limitation.

## v4.3.3 Run and Results

The v4.3.3 3000-sample run uses GPUs 4–7, 2400 updates, two autonomous training
steps, the validation-selected `remaining` schedule, inference motion scale
`0.5`, and CA loss weights `0.1` (displacement) and `0.05` (direction):

```bash
bash pocketdiff_v4/run_v433_3000.sh
bash pocketdiff_v4/evaluate_v433_controls.sh
```

Training completed for 200 epochs. Best validation autonomous backbone RMSD
was `0.42208 Å`. On validation, the selected two-step prediction improved
atom/CA/backbone RMSD by `0.01271/0.02015/0.01811 Å`. On test, atom RMSD
improved by `0.00414 Å` with paired-bootstrap 95% CI
`[-0.00090, 0.00972] Å`; CA and backbone gains were smaller and their
confidence intervals crossed zero. Test direction cosine increased from
`0.0641` in v4.3.2 to `0.1132`, while the second autonomous step partially
reversed the first-step improvement. Noisy learned controls still regressed
against noisy apo, so this model is not a denoiser.
