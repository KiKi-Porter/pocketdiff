"""Real 3txj PocketDiff protein-only single-step smoke."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.data.schema import ResidueMetadata
from pocketdiff.models import PocketDiffModel
from pocketdiff.sampling import pocket_step
from pocketdiff.targetdiff import initialize_targetdiff_state


def _find_record(records):
    for record in records:
        if str(record[0]).startswith("3txj") or str(record[1]).startswith("3txj"):
            return record
    raise RuntimeError("3txj record was not found in the Apo2Mol train split")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".codex-tasks/pocketdiff-development/phase6b-clean-generalization/raw/generalization_report.pt"),
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k < 0 or args.k > 19:
        parser.error("k must lie in [0, 19]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.split_pickle.open("rb") as handle:
        record = _find_record(pickle.load(handle)["train"])
    value = Apo2MolAdapter(args.data_root).convert_record(record)
    batch_protein = torch.zeros(value.num_protein_atoms, dtype=torch.long)
    batch_ligand = torch.zeros(value.num_ligand_atoms, dtype=torch.long)
    state = initialize_targetdiff_state(
        protein_pos=value.protein_pos_apo,
        protein_v=value.protein_feature,
        batch_protein=batch_protein,
        ligand_pos=value.ligand_pos_ref,
        ligand_v=value.ligand_type_ref,
        batch_ligand=batch_ligand,
        apo_pos_ref=value.protein_pos_apo,
        center_mode="protein",
    )
    metadata = ResidueMetadata(
        protein_atom_name=[list(value.protein_atom_name)],
        protein_residue_name=[list(value.protein_residue_name)],
        residue_type=value.residue_type,
        atom_to_residue_global=value.atom_to_residue,
        batch_residue=torch.zeros(value.num_residues, dtype=torch.long),
        frame_valid_reference=value.frame_valid,
        chi_mask=value.chi_mask,
        chain_id=[list(value.residue_chain_id)],
        residue_sequence_id=[list(value.residue_sequence_id)],
    )
    model = PocketDiffModel(encoder_layers=1, knn=8, sigma_translation=1.0)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    before = {
        name: getattr(state, name).clone()
        for name in ("protein_v", "batch_protein", "ligand_pos", "ligand_v", "batch_ligand", "apo_pos_ref", "center_offset")
    }
    output = pocket_step(model, state, metadata, k=args.k)
    report = {
        "sample_id": value.sample_id,
        "k": args.k,
        "targetdiff_t": 199 - 10 * args.k,
        "remaining_steps": 20 - args.k,
        "num_protein_atoms": value.num_protein_atoms,
        "num_residues": value.num_residues,
        "num_ligand_atoms": value.num_ligand_atoms,
        "valid_residues": int(output.prediction.frame_valid.sum().item()),
        "protein_finite": bool(torch.isfinite(output.protein_pos_next).all()),
        "prediction_finite": bool(torch.isfinite(output.prediction.remaining_translation_local).all() and torch.isfinite(output.prediction.remaining_rotvec_local).all()),
        "applied_finite": bool(torch.isfinite(output.applied_translation_local).all() and torch.isfinite(output.applied_rotvec_local).all()),
        "protein_changed": bool(not torch.equal(output.protein_pos_next, state.protein_pos)),
        "protein_feature_unchanged": bool(torch.equal(state.protein_v, before["protein_v"])),
        "batch_unchanged": bool(torch.equal(state.batch_protein, before["batch_protein"]) and torch.equal(state.batch_ligand, before["batch_ligand"])),
        "ligand_unchanged": bool(torch.equal(state.ligand_pos, before["ligand_pos"]) and torch.equal(state.ligand_v, before["ligand_v"])),
        "apo_reference_unchanged": bool(torch.equal(state.apo_pos_ref, before["apo_pos_ref"])),
        "center_offset_unchanged": bool(torch.equal(state.center_offset, before["center_offset"])),
        "max_atom_displacement": float(torch.linalg.vector_norm(output.protein_pos_next - state.protein_pos, dim=-1).max().item()),
    }
    required = (
        "protein_finite", "prediction_finite", "applied_finite", "protein_feature_unchanged",
        "batch_unchanged", "ligand_unchanged", "apo_reference_unchanged", "center_offset_unchanged",
    )
    if not all(report[key] for key in required):
        raise RuntimeError(f"Pocket step smoke failed: {report}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
