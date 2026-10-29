"""Run one independent PocketDiff MVP forward on a raw Apo2Mol sample."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.models import PocketDiffModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.split_pickle.open("rb") as handle:
        record = pickle.load(handle)["train"][0]
    complex_value, adapter_diagnostics = Apo2MolAdapter(args.data_root).convert_record_with_diagnostics(record)
    model = PocketDiffModel(encoder_layers=1, knn=8)
    model.eval()
    before_ligand_pos = complex_value.ligand_pos_ref.clone()
    before_ligand_type = complex_value.ligand_type_ref.clone()
    with torch.no_grad():
        prediction = model.forward_complex(complex_value, targetdiff_t=199, pocket_k=0)
    report = {
        "sample_id": complex_value.sample_id,
        "num_protein_atoms": complex_value.num_protein_atoms,
        "num_residues": complex_value.num_residues,
        "num_ligand_atoms": complex_value.num_ligand_atoms,
        "prediction_translation_shape": list(prediction.remaining_translation_local.shape),
        "prediction_rotvec_shape": list(prediction.remaining_rotvec_local.shape),
        "prediction_translation_abs_max": float(prediction.remaining_translation_local.abs().max()),
        "prediction_rotvec_abs_max": float(prediction.remaining_rotvec_local.abs().max()),
        "prediction_frame_valid": int(prediction.frame_valid.sum()),
        "prediction_frame_invalid": int((~prediction.frame_valid).sum()),
        "prediction_finite": bool(torch.isfinite(prediction.remaining_translation_local).all() and torch.isfinite(prediction.remaining_rotvec_local).all()),
        "ligand_pos_unchanged": bool(torch.equal(before_ligand_pos, complex_value.ligand_pos_ref)),
        "ligand_type_unchanged": bool(torch.equal(before_ligand_type, complex_value.ligand_type_ref)),
        "diagnostics": {name: float(value) for name, value in prediction.diagnostics.items()},
        "adapter": adapter_diagnostics.to_dict(),
        "encoder": "independent_distance_invariant_mvp",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("sample_id", "prediction_translation_shape", "prediction_rotvec_shape", "prediction_finite")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
