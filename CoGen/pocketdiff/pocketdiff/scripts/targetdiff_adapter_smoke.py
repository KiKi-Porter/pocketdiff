"""Real TargetDiff checkpoint and one-step adapter smoke on Apo2Mol 3txj."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.targetdiff import TargetDiffAdapter, initialize_targetdiff_state


def _find_record(records):
    for record in records:
        if str(record[0]).startswith("3txj") or str(record[1]).startswith("3txj"):
            return record
    raise RuntimeError("3txj record was not found in the Apo2Mol train split")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("targetdiff-main/targetdiff-main/pretrained_models/pretrained_diffusion.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.split_pickle.open("rb") as handle:
        record = _find_record(pickle.load(handle)["train"])
    value = Apo2MolAdapter(args.data_root).convert_record(record)
    zeros_protein = torch.zeros(value.num_protein_atoms, dtype=torch.long)
    zeros_ligand = torch.zeros(value.num_ligand_atoms, dtype=torch.long)
    state = initialize_targetdiff_state(
        protein_pos=value.protein_pos_apo,
        protein_v=value.protein_feature,
        batch_protein=zeros_protein,
        ligand_pos=value.ligand_pos_ref,
        ligand_v=value.ligand_type_ref,
        batch_ligand=zeros_ligand,
        apo_pos_ref=value.protein_pos_apo,
        center_mode="protein",
    )
    before = {
        "protein_pos": state.protein_pos.clone(),
        "protein_v": state.protein_v.clone(),
        "batch_protein": state.batch_protein.clone(),
        "batch_ligand": state.batch_ligand.clone(),
        "apo_pos_ref": state.apo_pos_ref.clone(),
    }
    adapter = TargetDiffAdapter.from_checkpoint(args.checkpoint, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(2021)
    next_state, aux = adapter.sample_step(state, 199, generator=generator)
    report = {
        "sample_id": value.sample_id,
        "t": 199,
        "num_graphs": state.num_graphs,
        "num_protein_atoms": int(state.protein_pos.shape[0]),
        "num_ligand_atoms": int(state.ligand_pos.shape[0]),
        "num_classes": adapter.num_classes,
        "num_timesteps": adapter.num_timesteps,
        "protein_finite": bool(torch.isfinite(next_state.protein_pos).all()),
        "ligand_position_finite": bool(torch.isfinite(next_state.ligand_pos).all()),
        "ligand_type_valid": bool(((next_state.ligand_v >= 0) & (next_state.ligand_v < 13)).all()),
        "pred_x0_finite": bool(torch.isfinite(aux.pred_x0).all()),
        "pred_v0_prob_finite": bool(torch.isfinite(aux.pred_v0_prob).all()),
        "posterior_v_prev_prob_finite": bool(torch.isfinite(aux.posterior_v_prev_prob).all()),
        "protein_unchanged": bool(torch.equal(next_state.protein_pos, before["protein_pos"])),
        "protein_features_unchanged": bool(torch.equal(next_state.protein_v, before["protein_v"])),
        "batch_unchanged": bool(torch.equal(next_state.batch_protein, before["batch_protein"]) and torch.equal(next_state.batch_ligand, before["batch_ligand"])),
        "apo_reference_unchanged": bool(torch.equal(next_state.apo_pos_ref, before["apo_pos_ref"])),
        "center_offset_finite": bool(torch.isfinite(next_state.center_offset).all()),
    }
    if not all(report[key] for key in (
        "protein_finite", "ligand_position_finite", "ligand_type_valid", "pred_x0_finite",
        "pred_v0_prob_finite", "posterior_v_prev_prob_finite", "protein_unchanged",
        "protein_features_unchanged", "batch_unchanged", "apo_reference_unchanged",
        "center_offset_finite",
    )):
        raise RuntimeError(f"TargetDiff adapter smoke failed: {report}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
