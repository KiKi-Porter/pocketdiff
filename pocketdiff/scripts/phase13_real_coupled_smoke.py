"""Real 3txj bounded/full coupled sampling smoke."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.data.schema import ResidueMetadata
from pocketdiff.evaluation import evaluate_protein_motion
from pocketdiff.models import PocketDiffModel
from pocketdiff.sampling import PocketStepSolver, run_coupled_sampling
from pocketdiff.targetdiff import TargetDiffAdapter, restore_center


def _find_record(records):
    for record in records:
        if str(record[0]).startswith("3txj") or str(record[1]).startswith("3txj"):
            return record
    raise RuntimeError("3txj record was not found in the Apo2Mol train split")


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


def _load_models(args):
    targetdiff = TargetDiffAdapter.from_checkpoint(args.targetdiff_checkpoint, device="cpu")
    pocket_payload = torch.load(args.pocketdiff_checkpoint, map_location="cpu")
    pocket_model = PocketDiffModel(**pocket_payload["model_config"])
    pocket_model.load_state_dict(pocket_payload["model_state_dict"], strict=True)
    pocket_model.eval()
    return targetdiff, PocketStepSolver(pocket_model)


def _make_initial_state(value, adapter, seed):
    batch_protein = torch.zeros(value.num_protein_atoms, dtype=torch.long)
    batch_ligand = torch.zeros(value.num_ligand_atoms, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = adapter.initialize_sampling_state(
        protein_pos=value.protein_pos_apo.float(),
        protein_v=value.protein_feature.float(),
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        apo_pos_ref=value.protein_pos_apo.float(),
        generator=generator,
    )
    return state, generator


def _finite_aux(auxiliaries):
    return bool(
        all(
            torch.isfinite(aux.pred_x0).all()
            and torch.isfinite(aux.pred_v0_prob).all()
            and torch.isfinite(aux.posterior_v_prev_prob).all()
            for aux in auxiliaries
        )
    )


def _bounded_report(value, adapter, state, generator, steps, elapsed):
    end_t = 1000 - steps
    final_state, auxiliaries = adapter.run_steps(
        state,
        t_start=999,
        t_end_inclusive=end_t,
        generator=generator,
    )
    return {
        "mode": "bounded",
        "full_result": False,
        "sample_id": value.sample_id,
        "steps": steps,
        "timesteps": list(range(999, end_t - 1, -1)),
        "num_protein_atoms": int(state.protein_pos.shape[0]),
        "num_ligand_atoms": int(state.ligand_pos.shape[0]),
        "finite": bool(torch.isfinite(final_state.ligand_pos).all() and _finite_aux(auxiliaries)),
        "protein_unchanged": bool(torch.equal(final_state.protein_pos, state.protein_pos)),
        "ligand_type_valid": bool(((final_state.ligand_v >= 0) & (final_state.ligand_v < 13)).all()),
        "elapsed_seconds": elapsed,
    }


def _full_report(value, state, result, elapsed):
    block_records = []
    previous_protein = state.protein_pos
    all_block_finite = True
    all_block_invariants = True
    for index, block in enumerate(result.blocks):
        block_finite = bool(
            torch.isfinite(block.state.protein_pos).all()
            and torch.isfinite(block.state.ligand_pos).all()
            and torch.isfinite(block.pocket_output.prediction.remaining_translation_local).all()
            and torch.isfinite(block.pocket_output.prediction.remaining_rotvec_local).all()
            and _finite_aux(block.targetdiff_aux)
        )
        block_invariants = bool(
            block.event.ligand_unchanged_during_pocket
            and block.event.protein_features_unchanged
            and block.event.apo_reference_unchanged
            and block.event.batch_unchanged
            and block.event.center_offset_unchanged
        )
        all_block_finite = all_block_finite and block_finite
        all_block_invariants = all_block_invariants and block_invariants
        block_records.append(
            {
                "k": index,
                "pocket_t": block.event.pocket_targetdiff_t,
                "targetdiff_timesteps": list(block.event.targetdiff_timesteps),
                "valid_residues": int(block.pocket_output.prediction.frame_valid.sum().item()),
                "max_pocket_atom_displacement": float(
                    torch.linalg.vector_norm(block.pocket_output.protein_pos_next - previous_protein, dim=-1).max().item()
                ),
                "finite": block_finite,
                "invariants": block_invariants,
            }
        )
        previous_protein = block.state.protein_pos
    final_state = result.state
    restored_protein = restore_center(final_state.protein_pos, final_state.batch_protein, final_state.center_offset)
    restored_ligand = restore_center(final_state.ligand_pos, final_state.batch_ligand, final_state.center_offset)
    geometry = evaluate_protein_motion(
        value.protein_pos_apo.float(),
        value.protein_pos_holo.float(),
        final_state.protein_pos,
        final_state.center_offset,
        final_state.batch_protein,
    )
    finite = {
        "protein": bool(torch.isfinite(final_state.protein_pos).all()),
        "ligand_position": bool(torch.isfinite(final_state.ligand_pos).all()),
        "restored_protein": bool(torch.isfinite(restored_protein).all()),
        "restored_ligand": bool(torch.isfinite(restored_ligand).all()),
        "prelude_aux": _finite_aux(result.prelude_aux),
        "blocks": all_block_finite,
        "ligand_type_valid": bool(((final_state.ligand_v >= 0) & (final_state.ligand_v < 13)).all()),
    }
    invariants = {
        "prelude_protein_unchanged": result.event.prelude_protein_unchanged,
        "protein_features_unchanged": result.event.protein_features_unchanged,
        "apo_reference_unchanged": result.event.apo_reference_unchanged,
        "batch_unchanged": result.event.batch_unchanged,
        "center_offset_unchanged": result.event.center_offset_unchanged,
        "block_invariants": all_block_invariants,
    }
    return {
        "mode": "full",
        "full_result": True,
        "sample_id": value.sample_id,
        "targetdiff_call_count": result.event.targetdiff_call_count,
        "pocket_call_count": result.event.pocket_call_count,
        "first_timestep": result.event.targetdiff_timesteps[0],
        "last_timestep": result.event.targetdiff_timesteps[-1],
        "pocket_timesteps": list(result.event.pocket_timesteps),
        "num_protein_atoms": int(final_state.protein_pos.shape[0]),
        "num_ligand_atoms": int(final_state.ligand_pos.shape[0]),
        "max_final_protein_displacement": float(torch.linalg.vector_norm(final_state.protein_pos - state.protein_pos, dim=-1).max().item()),
        "max_final_ligand_radius": float(torch.linalg.vector_norm(final_state.ligand_pos, dim=-1).max().item()),
        "finite": finite,
        "invariants": invariants,
        "geometry": geometry.as_dict(),
        "blocks": block_records,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--targetdiff-checkpoint", type=Path, default=Path("targetdiff-main/targetdiff-main/pretrained_models/pretrained_diffusion.pt"))
    parser.add_argument("--pocketdiff-checkpoint", type=Path, default=Path(".codex-tasks/pocketdiff-development/phase6b-clean-generalization/raw/generalization_report.pt"))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--bounded-steps", type=int, default=0, help="run only N initial TargetDiff steps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bounded_steps < 0 or args.bounded_steps > 1000:
        parser.error("bounded-steps must lie in [0, 1000]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.split_pickle.open("rb") as handle:
        record = _find_record(pickle.load(handle)["train"])
    value = Apo2MolAdapter(args.data_root).convert_record(record)
    adapter, pocket_solver = _load_models(args)
    state, generator = _make_initial_state(value, adapter, args.seed)
    metadata = _make_metadata(value)
    try:
        if args.bounded_steps:
            report = _bounded_report(value, adapter, state, generator, args.bounded_steps, time.monotonic() - started)
            report["passed"] = bool(report["finite"] and report["protein_unchanged"] and report["ligand_type_valid"])
        else:
            result = run_coupled_sampling(adapter, pocket_solver, state, metadata, generator=generator)
            report = _full_report(value, state, result, time.monotonic() - started)
            report["passed"] = bool(all(report["finite"].values()) and all(report["invariants"].values()))
    except Exception as exc:
        report = {
            "mode": "full" if not args.bounded_steps else "bounded",
            "full_result": not bool(args.bounded_steps),
            "sample_id": value.sample_id,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        }
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        raise
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError(f"real coupled smoke failed: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
