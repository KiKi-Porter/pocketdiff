"""Run the first official TargetDiff/PocketDiff late-stage bridge experiment.

The experiment compares two conditions on the same ten test complexes:

* TargetDiff-only: fixed apo protein for all 1000 reverse steps.
* Coupled: fixed apo protein through t=200, then one v48 PocketDiff update
  before each ten-step TargetDiff block at t=199, 189, ..., 9.

The TargetDiff random trajectory is replayed from an explicit RNG trace so
the comparison differs only in the protein update condition.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import torch

from pocketdiff.diffusion import apply_local_se3_update, build_diffusion_state_from_current
from pocketdiff.diffusion.model import DiffusionMotionAdapter
from pocketdiff.diffusion.state import DiffusionState
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.preprocessing.diffusion_cache import load_diffusion_cache
from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffRNGTrace,
    TargetDiffStepRandomness,
)
from pocketdiff.targetdiff.state import TargetDiffState
from pocketdiff.training.diffusion import load_diffusion_checkpoint


DEFAULT_SAMPLE_IDS = (
    "9f5f__1__1.A__1.B",
    "8t7q__2__1.B__1.D",
    "8tn6__1__1.A__1.D",
    "7g01__1__1.A__1.D",
    "7g0m__1__1.A__1.B",
    "8uzo__1__1.C__1.I",
    "8rdk__1__1.A__1.D",
    "8puy__1__1.A__1.C",
    "7g12__1__1.A__1.B",
    "7fzl__1__1.A__1.B",
)

_LIGAND_ELEMENTS = ("C", "N", "O", "F", "P", "S", "CL", "BR", "I")
_PROTEIN_ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "CL", 34: "SE"}


def _pdb_atom(
    record: str,
    serial: int,
    name: str,
    residue: str,
    chain: str,
    residue_id: str,
    position: Sequence[float],
    element: str,
) -> str:
    try:
        residue_number = int(str(residue_id).strip())
    except ValueError:
        residue_number = 1
    atom_name = str(name).strip()[:4].rjust(4)
    return (
        f"{record:<6}{serial:5d} {atom_name} {residue[:3]:>3} "
        f"{(chain or 'A')[:1]:1}{residue_number:4d}    "
        f"{float(position[0]):8.3f}{float(position[1]):8.3f}"
        f"{float(position[2]):8.3f}{1.00:6.2f}{0.00:6.2f}"
        f"          {element:>2}\n"
    )


def _write_multimodel_pdb(
    path: Path,
    value,
    protein_states: Sequence[torch.Tensor],
    ligand_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    center: torch.Tensor,
) -> None:
    if len(protein_states) != len(ligand_states):
        raise ValueError("protein and ligand trajectory lengths differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    residue_chain = [
        value.residue_chain_id[int(residue_id)]
        for residue_id in value.atom_to_residue.detach().cpu().tolist()
    ]
    residue_sequence = [
        value.residue_sequence_id[int(residue_id)]
        for residue_id in value.atom_to_residue.detach().cpu().tolist()
    ]
    protein_elements = [
        _PROTEIN_ELEMENTS.get(int(number), "C")
        for number in value.protein_element.detach().cpu().tolist()
    ]
    with path.open("w") as handle:
        for model_index, (protein, ligand) in enumerate(
            zip(protein_states, ligand_states), start=1
        ):
            handle.write(f"MODEL     {model_index:4d}\n")
            protein_positions = (protein.detach().cpu() + center.cpu()).tolist()
            for serial, row in enumerate(
                zip(
                    value.protein_atom_name,
                    value.protein_residue_name,
                    residue_chain,
                    residue_sequence,
                    protein_elements,
                    protein_positions,
                ),
                start=1,
            ):
                name, residue, chain, residue_id, element, position = row
                handle.write(
                    _pdb_atom(
                        "ATOM", serial, name, residue, chain, residue_id, position, element
                    )
                )
            ligand_pos, ligand_type = ligand
            ligand_positions = (ligand_pos.detach().cpu() + center.cpu()).tolist()
            for offset, (atom_type, position) in enumerate(
                zip(ligand_type.detach().cpu().tolist(), ligand_positions), start=1
            ):
                element = _LIGAND_ELEMENTS[int(atom_type) % len(_LIGAND_ELEMENTS)]
                handle.write(
                    _pdb_atom(
                        "HETATM",
                        len(protein_positions) + offset,
                        f"L{offset}",
                        "LIG",
                        "Z",
                        "1",
                        position,
                        element,
                    )
                )
            handle.write("TER\nENDMDL\n")
        handle.write("END\n")


def _centered_value(value):
    center = value.protein_pos_apo.mean(dim=0)
    return (
        replace(
            value,
            protein_pos_apo=value.protein_pos_apo - center,
            protein_pos_holo=value.protein_pos_holo - center,
            ligand_pos_ref=value.ligand_pos_ref - center,
            center_offset=torch.zeros_like(value.center_offset),
        ),
        center,
    )


def _move_value(value, device: torch.device):
    fields = {}
    for field in value.__dataclass_fields__.values():
        item = getattr(value, field.name)
        fields[field.name] = item.to(device) if isinstance(item, torch.Tensor) else item
    return type(value)(**fields)


def _rng_trace(adapter: TargetDiffAdapter, ligand_count: int, seed: int, device: torch.device):
    generator = torch.Generator(device=device.type).manual_seed(int(seed))
    steps = {}
    for timestep in range(adapter.num_timesteps):
        steps[timestep] = TargetDiffStepRandomness(
            position_noise=torch.randn(
                (ligand_count, 3), generator=generator, device=device
            ),
            categorical_uniform=torch.rand(
                (ligand_count, adapter.num_classes),
                generator=generator,
                device=device,
            ),
        )
    return TargetDiffRNGTrace(steps)


def _initial_targetdiff_state(adapter, value, seed: int) -> TargetDiffState:
    protein_pos = value.protein_pos_apo.float()
    batch_protein = torch.zeros(
        value.num_protein_atoms, dtype=torch.long, device=protein_pos.device
    )
    batch_ligand = torch.zeros(
        value.num_ligand_atoms, dtype=torch.long, device=protein_pos.device
    )
    generator = torch.Generator(device=protein_pos.device.type).manual_seed(int(seed))
    return adapter.initialize_sampling_state(
        protein_pos=protein_pos,
        protein_v=value.protein_feature.float(),
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        apo_pos_ref=protein_pos,
        generator=generator,
    )


def _run_targetdiff_only(
    adapter: TargetDiffAdapter,
    initial: TargetDiffState,
    trace: TargetDiffRNGTrace,
):
    prelude, _ = adapter.run_steps(
        initial, t_start=999, t_end_inclusive=200, rng_trace=trace
    )
    protein_states = [prelude.protein_pos.clone()]
    ligand_states = [(prelude.ligand_pos.clone(), prelude.ligand_v.clone())]
    current = prelude
    for start_t in range(199, -1, -10):
        current, _ = adapter.run_steps(
            current,
            t_start=start_t,
            t_end_inclusive=start_t - 9,
            rng_trace=trace,
        )
        protein_states.append(current.protein_pos.clone())
        ligand_states.append((current.ligand_pos.clone(), current.ligand_v.clone()))
    return current, protein_states, ligand_states


def _make_pocket_state(value, current: TargetDiffState, t: float) -> DiffusionState:
    # The generated ligand has a TargetDiff-dependent atom count/type set.
    # Replacing the cached reference ligand on a centered copy preserves all
    # protein topology and holo supervision while satisfying the v48 state
    # contract.
    bridge_value = replace(
        value,
        ligand_pos_ref=current.ligand_pos,
        ligand_type_ref=current.ligand_v,
    )
    return build_diffusion_state_from_current(
        bridge_value,
        current.protein_pos,
        float(t),
        ligand_pos=current.ligand_pos,
        ligand_type=current.ligand_v,
        ligand_condition_t=199 - int(round((1.0 - float(t)) * 200.0)),
        target_parameterization="bridge_velocity",
        target_steps=20,
    )


@torch.no_grad()
def _pocket_update(
    model: DiffusionMotionAdapter,
    value,
    current: TargetDiffState,
    pocket_index: int,
) -> tuple[TargetDiffState, dict]:
    t = 1.0 - float(pocket_index) / 20.0
    diffusion_state = _make_pocket_state(value, current, t)
    prediction = model(diffusion_state.model_input)
    input_value = diffusion_state.model_input
    frames = build_residue_frames(
        current.protein_pos,
        input_value.atom_to_residue,
        input_value.protein_atom_name,
        num_residues=input_value.residue_type.shape[0],
        atom_indices=input_value.frame_atom_indices,
    )
    valid = input_value.frame_valid & frames.valid
    protein_next = apply_local_se3_update(
        current.protein_pos,
        input_value.atom_to_residue,
        frames.origins,
        frames.frames,
        prediction.translation_local,
        prediction.rotation_local,
        fraction=1.0 / 20.0,
        frame_valid=valid,
    )
    displacement = torch.linalg.vector_norm(
        protein_next - current.protein_pos, dim=-1
    )
    return current.replace(protein_pos=protein_next), {
        "pocket_index": pocket_index,
        "targetdiff_t": 199 - 10 * pocket_index,
        "pocket_time": t,
        "valid_residue_count": int(valid.sum().item()),
        "max_atom_displacement": float(displacement.max().item()),
        "mean_atom_displacement": float(displacement.mean().item()),
        "prediction_finite": bool(
            torch.isfinite(prediction.translation_local).all()
            and torch.isfinite(prediction.rotation_local).all()
        ),
        "protein_changed": bool(not torch.equal(protein_next, current.protein_pos)),
    }


def _run_coupled(
    adapter: TargetDiffAdapter,
    pocket_model: DiffusionMotionAdapter,
    value,
    initial: TargetDiffState,
    trace: TargetDiffRNGTrace,
):
    prelude, _ = adapter.run_steps(
        initial, t_start=999, t_end_inclusive=200, rng_trace=trace
    )
    protein_states = [prelude.protein_pos.clone()]
    ligand_states = [(prelude.ligand_pos.clone(), prelude.ligand_v.clone())]
    diagnostics = []
    current = prelude
    for pocket_index, start_t in enumerate(range(199, -1, -10)):
        current, pocket_diag = _pocket_update(
            pocket_model, value, current, pocket_index
        )
        diagnostics.append(pocket_diag)
        current, _ = adapter.run_steps(
            current,
            t_start=start_t,
            t_end_inclusive=start_t - 9,
            rng_trace=trace,
        )
        protein_states.append(current.protein_pos.clone())
        ligand_states.append((current.ligand_pos.clone(), current.ligand_v.clone()))
    return current, protein_states, ligand_states, diagnostics


def _summary(value, targetdiff_only, coupled, coupled_diag):
    final_protein = coupled.protein_pos
    apo = value.protein_pos_apo
    holo = value.protein_pos_holo
    backbone_mask = torch.tensor(
        [name in {"N", "CA", "C", "O"} for name in value.protein_atom_name],
        dtype=torch.bool,
        device=apo.device,
    )
    def rmsd(a, b):
        return float(torch.sqrt(((a - b).square()).sum(dim=-1).mean()).item())
    return {
        "sample_id": value.sample_id,
        "ligand_atoms_targetdiff_only": int(targetdiff_only.ligand_pos.shape[0]),
        "ligand_atoms_coupled": int(coupled.ligand_pos.shape[0]),
        "targetdiff_final_ligand_finite": bool(torch.isfinite(targetdiff_only.ligand_pos).all()),
        "coupled_final_ligand_finite": bool(torch.isfinite(coupled.ligand_pos).all()),
        "coupled_protein_finite": bool(torch.isfinite(final_protein).all()),
        "coupled_final_protein_displacement_max": float(
            torch.linalg.vector_norm(final_protein - apo, dim=-1).max().item()
        ),
        "coupled_final_protein_displacement_mean": float(
            torch.linalg.vector_norm(final_protein - apo, dim=-1).mean().item()
        ),
        "apo_backbone_rmsd_to_holo": rmsd(apo[backbone_mask], holo[backbone_mask]),
        "coupled_backbone_rmsd_to_holo": rmsd(
            final_protein[backbone_mask], holo[backbone_mask]
        ),
        "pocket_calls": len(coupled_diag),
        "pocket_changed_calls": int(sum(item["protein_changed"] for item in coupled_diag)),
        "max_pocket_atom_displacement": max(
            item["max_atom_displacement"] for item in coupled_diag
        ),
    }


def run(args) -> int:
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = load_diffusion_cache(Path(args.prepared_cache))
    by_id = {
        item.sample_id: item
        for item in cache["splits"]["test"]
    }
    sample_ids = tuple(
        item.strip() for item in args.sample_ids.split(",") if item.strip()
    )
    missing = [item for item in sample_ids if item not in by_id]
    if missing:
        raise ValueError(f"sample ids are missing from test cache: {missing}")
    targetdiff = TargetDiffAdapter.from_checkpoint(
        args.targetdiff_checkpoint, device=device
    )
    loaded = load_diffusion_checkpoint(args.pocketdiff_checkpoint, map_location=device)
    pocket_model = loaded.model
    if pocket_model.prediction_type != "bridge_velocity":
        raise ValueError(
            f"bridge requires v48 bridge_velocity checkpoint, got {pocket_model.prediction_type}"
        )
    pocket_model.eval()
    config = {
        "version": "bridge-phase1-20261001-v48",
        "prepared_cache": str(Path(args.prepared_cache).resolve()),
        "targetdiff_checkpoint": str(Path(args.targetdiff_checkpoint).resolve()),
        "pocketdiff_checkpoint": str(Path(args.pocketdiff_checkpoint).resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "sample_ids": list(sample_ids),
        "targetdiff_schedule": "999->0",
        "prelude": "999->200",
        "late_blocks": "199..190,...,9..0",
        "pocketdiff_calls": 20,
        "pocketdiff_fraction": 1.0 / 20.0,
        "holo_is_model_input": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if args.smoke:
        sample_ids = sample_ids[:1]
    records = []
    started = time.monotonic()
    for index, sample_id in enumerate(sample_ids):
        raw_value = by_id[sample_id]
        value, center = _centered_value(raw_value)
        value = _move_value(value, device)
        sample_seed = int(args.seed) + index
        initial = _initial_targetdiff_state(targetdiff, value, sample_seed)
        trace = _rng_trace(
            targetdiff, initial.ligand_pos.shape[0], sample_seed + 100000, device
        )
        targetdiff_result = _run_targetdiff_only(targetdiff, initial, trace)
        coupled_result = _run_coupled(
            targetdiff, pocket_model, value, initial, trace
        )
        targetdiff_final, targetdiff_protein, targetdiff_ligand = targetdiff_result
        coupled_final, coupled_protein, coupled_ligand, diagnostics = coupled_result
        sample_dir = output / "samples" / f"{index:02d}_{sample_id.replace('.', '-')}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write_multimodel_pdb(
            sample_dir / "targetdiff_only_trajectory.pdb",
            raw_value,
            targetdiff_protein,
            targetdiff_ligand,
            center,
        )
        _write_multimodel_pdb(
            sample_dir / "coupled_late200_trajectory.pdb",
            raw_value,
            coupled_protein,
            coupled_ligand,
            center,
        )
        torch.save(
            {
                "sample_id": sample_id,
                "targetdiff_only": {
                    "protein_states": targetdiff_protein,
                    "ligand_states": targetdiff_ligand,
                },
                "coupled": {
                    "protein_states": coupled_protein,
                    "ligand_states": coupled_ligand,
                    "pocket_diagnostics": diagnostics,
                },
            },
            sample_dir / "trajectory.pt",
        )
        record = _summary(value, targetdiff_final, coupled_final, diagnostics)
        record["sample_seed"] = sample_seed
        record["targetdiff_trajectory_pdb"] = str(
            (sample_dir / "targetdiff_only_trajectory.pdb").resolve()
        )
        record["coupled_trajectory_pdb"] = str(
            (sample_dir / "coupled_late200_trajectory.pdb").resolve()
        )
        record["diagnostics"] = diagnostics
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.smoke:
            break

    report = {
        "config": config,
        "elapsed_seconds": time.monotonic() - started,
        "sample_count": len(records),
        "records": records,
        "all_finite": bool(
            records
            and all(
                item["targetdiff_final_ligand_finite"]
                and item["coupled_final_ligand_finite"]
                and item["coupled_protein_finite"]
                for item in records
            )
        ),
        "all_coupled_calls_changed_protein": bool(
            records and all(item["pocket_changed_calls"] > 0 for item in records)
        ),
    }
    (output / ("smoke_report.json" if args.smoke else "comparison.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0 if report["all_finite"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-cache", type=Path, required=True)
    parser.add_argument("--targetdiff-checkpoint", type=Path, required=True)
    parser.add_argument("--pocketdiff-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--sample-ids", default=",".join(DEFAULT_SAMPLE_IDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.smoke == args.full:
        parser.error("choose exactly one of --smoke or --full")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
