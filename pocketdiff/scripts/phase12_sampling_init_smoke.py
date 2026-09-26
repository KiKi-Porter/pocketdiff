"""Real 3txj smoke for official-style initial ligand state generation."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.targetdiff import TargetDiffAdapter, restore_center


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
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.split_pickle.open("rb") as handle:
        record = _find_record(pickle.load(handle)["train"])
    value = Apo2MolAdapter(args.data_root).convert_record(record)
    adapter = TargetDiffAdapter.from_checkpoint(args.checkpoint, device="cpu")
    batch_protein = torch.zeros(value.num_protein_atoms, dtype=torch.long)
    batch_ligand = torch.zeros(value.num_ligand_atoms, dtype=torch.long)
    before = {
        "protein_pos": value.protein_pos_apo.clone(),
        "protein_v": value.protein_feature.clone(),
        "batch_protein": batch_protein.clone(),
        "batch_ligand": batch_ligand.clone(),
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    state = adapter.initialize_sampling_state(
        protein_pos=value.protein_pos_apo.float(),
        protein_v=value.protein_feature.float(),
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        apo_pos_ref=value.protein_pos_apo.float(),
        generator=generator,
    )
    after_init = {
        name: getattr(state, name).clone()
        for name in ("protein_pos", "protein_v", "batch_protein", "batch_ligand", "apo_pos_ref", "center_offset")
    }
    final_state, auxiliaries = adapter.run_steps(
        state,
        t_start=999,
        t_end_inclusive=996,
        generator=generator,
    )
    raw_ligand_pos = restore_center(state.ligand_pos, state.batch_ligand, state.center_offset)
    finite = {
        "initial_ligand_position": bool(torch.isfinite(state.ligand_pos).all()),
        "initial_ligand_type": bool(((state.ligand_v >= 0) & (state.ligand_v < 13)).all()),
        "four_step_ligand_position": bool(torch.isfinite(final_state.ligand_pos).all()),
        "auxiliary_outputs": bool(
            all(torch.isfinite(aux.pred_x0).all() and torch.isfinite(aux.pred_v0_prob).all() and torch.isfinite(aux.posterior_v_prev_prob).all() for aux in auxiliaries)
        ),
        "raw_position_finite": bool(torch.isfinite(raw_ligand_pos).all()),
    }
    invariants = {
        "protein_unchanged": bool(torch.equal(final_state.protein_pos, state.protein_pos)),
        "protein_features_unchanged": bool(torch.equal(final_state.protein_v, state.protein_v)),
        "apo_reference_unchanged": bool(torch.equal(final_state.apo_pos_ref, state.apo_pos_ref)),
        "batch_unchanged": bool(torch.equal(final_state.batch_protein, state.batch_protein) and torch.equal(final_state.batch_ligand, state.batch_ligand)),
        "center_offset_unchanged": bool(torch.equal(final_state.center_offset, state.center_offset)),
        "input_arrays_not_mutated": bool(
            torch.equal(value.protein_pos_apo, before["protein_pos"])
            and torch.equal(value.protein_feature, before["protein_v"])
            and torch.equal(batch_protein, before["batch_protein"])
            and torch.equal(batch_ligand, before["batch_ligand"])
        ),
        "centered_protein_mean_zero": bool(torch.allclose(state.protein_pos.mean(dim=0), torch.zeros(3), atol=1e-6)),
        "state_init_metadata_stable": bool(
            torch.equal(state.protein_v, after_init["protein_v"])
            and torch.equal(state.batch_protein, after_init["batch_protein"])
            and torch.equal(state.batch_ligand, after_init["batch_ligand"])
            and torch.equal(state.apo_pos_ref, after_init["apo_pos_ref"])
            and torch.equal(state.center_offset, after_init["center_offset"])
        ),
    }
    report = {
        "sample_id": value.sample_id,
        "seed": args.seed,
        "timesteps": [999, 998, 997, 996],
        "num_protein_atoms": value.num_protein_atoms,
        "num_ligand_atoms": value.num_ligand_atoms,
        "num_classes": adapter.num_classes,
        "finite": finite,
        "invariants": invariants,
        "initial_ligand_position_mean_norm": float(torch.linalg.vector_norm(state.ligand_pos, dim=-1).mean().item()),
        "initial_ligand_type_min": int(state.ligand_v.min().item()),
        "initial_ligand_type_max": int(state.ligand_v.max().item()),
    }
    report["passed"] = bool(all(finite.values()) and all(invariants.values()))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError(f"sampling initialization smoke failed: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
