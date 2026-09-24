# PocketDiff v4

This repository snapshot contains the PocketDiff v4 residue-level baseline code.
The implementation is isolated under `pocketdiff_v4/`, with the minimal
`pocketdiff/` geometry/data helpers required by the v4 trainer and tests.

Large generated artifacts are intentionally excluded from Git:

- cached `.pt` graph datasets
- training checkpoints
- run logs and evaluation JSON files

The v4 model is a DynamicBind-style engineering prototype with residue-level
scalar/vector message passing, ligand-residue cross messages, local SE(3)
motion heads, sparse chi updates, multi-step rollout training, and apo-only
sampling/evaluation.

Run the unit tests from the repository root with:

```bash
PYTHONPATH=. conda run -n targetdiff python -m pytest -q pocketdiff_v4/tests
```

See `pocketdiff_v4/README.md` for architecture details, training commands, and
the latest baseline decision.
