"""Real-data smoke for the Phase 28 χ batch/head/loss contract."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import collate_clean_examples, make_clean_example, masked_periodic_chi_loss


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(".codex-tasks/pocketdiff-development")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "phase6b-clean-generalization/raw/manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["entries"][:args.limit]
    adapter = Apo2MolAdapter(".")
    examples = []
    source_ids = []
    for entry in entries:
        source = entry["source"]
        value, _diagnostics = adapter.convert_paths(
            source["holo_pocket"]["path"],
            source["apo_pocket"]["path"],
            source["ligand"]["path"],
            sample_id=entry["sample_id"],
        )
        examples.append(make_clean_example(value))
        source_ids.append(value.sample_id)

    batch = collate_clean_examples(examples)
    model = PocketDiffModel(encoder_layers=1, knn=8, predict_chi=True)
    with torch.no_grad():
        prediction = model(**batch.model_kwargs())
        chi_loss = masked_periodic_chi_loss(
            prediction, batch.chi_apo, batch.chi_holo, batch.chi_mask
        )
    if prediction.remaining_chi is None:
        raise RuntimeError("predict_chi=True returned no chi output")
    if not torch.isfinite(prediction.remaining_chi).all() or not torch.isfinite(chi_loss.loss):
        raise RuntimeError("non-finite chi output or loss")
    report = {
        "passed": True,
        "mode": "phase28_real_chi_head_contract_smoke",
        "sample_count": len(examples),
        "sample_ids": source_ids,
        "protein_residue_count": int(batch.residue_type.numel()),
        "chi_shape": list(batch.chi_apo.shape),
        "valid_chi_count": int(batch.chi_mask.sum()),
        "valid_residue_count": int(batch.chi_mask.any(dim=1).sum()),
        "prediction_shape": list(prediction.remaining_chi.shape),
        "prediction_abs_max": float(prediction.remaining_chi.abs().max()),
        "initial_periodic_chi_loss": float(chi_loss.loss),
        "source_manifest_sha256": _sha(args.manifest),
        "scope": "real adapter/batch/head/loss only; no optimizer step, cache rewrite or rollout",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "sample_count": report["sample_count"],
        "valid_chi_count": report["valid_chi_count"],
        "prediction_shape": report["prediction_shape"],
    }), flush=True)


if __name__ == "__main__":
    main()
