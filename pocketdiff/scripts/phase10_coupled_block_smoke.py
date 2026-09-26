"""Real 3txj smoke for one PocketDiff-first TargetDiff ten-step block."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.data.schema import ResidueMetadata
from pocketdiff.models import PocketDiffModel
from pocketdiff.sampling import PocketStepSolver, run_pocket_block
from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffRNGTrace,
    TargetDiffStepRandomness,
    initialize_targetdiff_state,
)


def _find_record(records):
    for record in records:
        if str(record[0]).startswith("3txj") or str(record[1]).startswith("3txj"):
            return record
    raise RuntimeError("3txj record was not found in the Apo2Mol train split")


def _make_state(value):
    batch_protein = torch.zeros(value.num_protein_atoms, dtype=torch.long)
    batch_ligand = torch.zeros(value.num_ligand_atoms, dtype=torch.long)
    return initialize_targetdiff_state(
        protein_pos=value.protein_pos_apo.float(),
        protein_v=value.protein_feature.float(),
        batch_protein=batch_protein,
        ligand_pos=value.ligand_pos_ref.float(),
        ligand_v=value.ligand_type_ref.long(),
        batch_ligand=batch_ligand,
        apo_pos_ref=value.protein_pos_apo.float(),
        center_mode="protein",
    )


def _make_metadata(value):
    return ResidueMetadata(
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


def _make_trace(state, start_t, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    timesteps = tuple(range(start_t, start_t - 10, -1))
    return TargetDiffRNGTrace(
        {
            timestep: TargetDiffStepRandomness(
                position_noise=torch.randn(state.ligand_pos.shape, generator=generator),
                categorical_uniform=torch.rand((state.ligand_pos.shape[0], 13), generator=generator),
            )
            for timestep in timesteps
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--targetdiff-checkpoint", type=Path, default=Path("targetdiff-main/targetdiff-main/pretrained_models/pretrained_diffusion.pt"))
    parser.add_argument("--pocketdiff-checkpoint", type=Path, default=Path(".codex-tasks/pocketdiff-development/phase6b-clean-generalization/raw/generalization_report.pt"))
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k < 0 or args.k > 19:
        parser.error("k must lie in [0, 19]")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.split_pickle.open("rb") as handle:
        record = _find_record(pickle.load(handle)["train"])
    value = Apo2MolAdapter(args.data_root).convert_record(record)
    state = _make_state(value)
    metadata = _make_metadata(value)

    pocket_checkpoint = torch.load(args.pocketdiff_checkpoint, map_location="cpu")
    pocket_model = PocketDiffModel(**pocket_checkpoint["model_config"])
    pocket_model.load_state_dict(pocket_checkpoint["model_state_dict"], strict=True)
    pocket_model.eval()
    pocket_solver = PocketStepSolver(pocket_model)
    targetdiff_adapter = TargetDiffAdapter.from_checkpoint(args.targetdiff_checkpoint, device="cpu")

    start_t = 199 - 10 * args.k
    trace = _make_trace(state, start_t, args.seed)
    before = {
        name: getattr(state, name).clone()
        for name in (
            "protein_pos",
            "protein_v",
            "batch_protein",
            "batch_ligand",
            "apo_pos_ref",
            "center_offset",
            "ligand_pos",
            "ligand_v",
        )
    }
    result = run_pocket_block(
        targetdiff_adapter,
        pocket_solver,
        state,
        metadata,
        args.k,
        rng_trace=trace,
    )
    output_state = result.state
    event = result.event
    finite = {
        "protein": bool(torch.isfinite(output_state.protein_pos).all()),
        "ligand_position": bool(torch.isfinite(output_state.ligand_pos).all()),
        "pocket_translation": bool(torch.isfinite(result.pocket_output.prediction.remaining_translation_local).all()),
        "pocket_rotvec": bool(torch.isfinite(result.pocket_output.prediction.remaining_rotvec_local).all()),
        "targetdiff_aux": bool(
            all(torch.isfinite(aux.pred_x0).all() and torch.isfinite(aux.pred_v0_prob).all() and torch.isfinite(aux.posterior_v_prev_prob).all() for aux in result.targetdiff_aux)
        ),
    }
    invariant_checks = {
        "input_state_not_mutated": bool(
            all(torch.equal(getattr(state, name), value_before) for name, value_before in before.items())
        ),
        "protein_features_unchanged": event.protein_features_unchanged,
        "apo_reference_unchanged": event.apo_reference_unchanged,
        "batch_unchanged": event.batch_unchanged,
        "center_offset_unchanged": event.center_offset_unchanged,
        "ligand_unchanged_during_pocket": event.ligand_unchanged_during_pocket,
        "ligand_types_valid": bool(((output_state.ligand_v >= 0) & (output_state.ligand_v < 13)).all()),
    }
    report = {
        "sample_id": value.sample_id,
        "k": args.k,
        "pocket_targetdiff_t": event.pocket_targetdiff_t,
        "targetdiff_timesteps": list(event.targetdiff_timesteps),
        "seed": args.seed,
        "num_protein_atoms": value.num_protein_atoms,
        "num_residues": value.num_residues,
        "num_ligand_atoms": value.num_ligand_atoms,
        "valid_residues": int(result.pocket_output.prediction.frame_valid.sum().item()),
        "valid_residue_fraction": float(result.pocket_output.prediction.frame_valid.float().mean().item()),
        "max_pocket_atom_displacement": float(torch.linalg.vector_norm(result.pocket_output.protein_pos_next - state.protein_pos, dim=-1).max().item()),
        "max_block_atom_displacement": float(torch.linalg.vector_norm(output_state.protein_pos - state.protein_pos, dim=-1).max().item()),
        "finite": finite,
        "invariant_checks": invariant_checks,
        "event": {
            "protein_changed_by_pocket": event.protein_changed_by_pocket,
            "protein_checksum_before": event.protein_checksum_before,
            "protein_checksum_after_pocket": event.protein_checksum_after_pocket,
            "protein_checksum_after_block": event.protein_checksum_after_block,
            "ligand_checksum_before": event.ligand_checksum_before,
            "ligand_checksum_after_pocket": event.ligand_checksum_after_pocket,
            "ligand_checksum_after_block": event.ligand_checksum_after_block,
        },
    }
    report["passed"] = bool(all(finite.values()) and all(invariant_checks.values()))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError(f"coupled block smoke failed: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
