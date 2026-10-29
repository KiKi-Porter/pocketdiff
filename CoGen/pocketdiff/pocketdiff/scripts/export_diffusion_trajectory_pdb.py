"""Export three deterministic PocketDiff trajectories as multi-model PDB files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from dataclasses import replace

from pocketdiff.training.diffusion import (
    DiffusionTrainConfig,
    _move_complex_to_device,
    load_apo2mol_slices,
    load_diffusion_checkpoint,
)
from pocketdiff.diffusion import sample_diffusion_state, sample_reverse_trajectory


_ELEMENTS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "CL",
    34: "SE",
}
_LIGAND_ELEMENTS = ("C", "N", "O", "F", "P", "S", "CL", "BR", "I")


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
    chain = (chain or "A")[:1]
    atom_name = str(name).strip()[:4]
    if len(atom_name) < 4:
        atom_name = atom_name.rjust(4)
    return (
        f"{record:<6}{serial:5d} {atom_name} {residue[:3]:>3} {chain:1}"
        f"{residue_number:4d}    {float(position[0]):8.3f}"
        f"{float(position[1]):8.3f}{float(position[2]):8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2}\n"
    )


def _write_multimodel_pdb(path: Path, value, states: Iterable[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residue_chain = [
        value.residue_chain_id[int(residue_id)]
        for residue_id in value.atom_to_residue.detach().cpu().tolist()
    ]
    residue_sequence = [
        value.residue_sequence_id[int(residue_id)]
        for residue_id in value.atom_to_residue.detach().cpu().tolist()
    ]
    atom_elements = [
        _ELEMENTS.get(int(number), "C")
        for number in value.protein_element.detach().cpu().tolist()
    ]
    ligand_types = value.ligand_type_ref.detach().cpu().tolist()
    with path.open("w") as handle:
        for model_index, protein_positions in enumerate(states, start=1):
            handle.write(f"MODEL     {model_index:4d}\n")
            positions = protein_positions.detach().cpu().tolist()
            for serial, (name, residue, chain, residue_id, element, position) in enumerate(
                zip(
                    value.protein_atom_name,
                    value.protein_residue_name,
                    residue_chain,
                    residue_sequence,
                    atom_elements,
                    positions,
                ),
                start=1,
            ):
                handle.write(
                    _pdb_atom(
                        "ATOM",
                        serial,
                        name,
                        residue,
                        chain,
                        residue_id,
                        position,
                        element,
                    )
                )
            ligand_positions = value.ligand_pos_ref.detach().cpu().tolist()
            for offset, (ligand_type, position) in enumerate(
                zip(ligand_types, ligand_positions),
                start=1,
            ):
                element = _LIGAND_ELEMENTS[
                    int(ligand_type) % len(_LIGAND_ELEMENTS)
                ]
                handle.write(
                    _pdb_atom(
                        "HETATM",
                        len(positions) + offset,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--train-count", type=int, default=20195)
    parser.add_argument("--valid-count", type=int, default=985)
    parser.add_argument("--test-count", type=int, default=450)
    parser.add_argument("--holdout-count", type=int, default=985)
    parser.add_argument(
        "--sample-ids",
        default="8v1p__1__1.A__1.B,8poj__2__1.B__1.D,8k5w__1__1.A__1.H",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    loaded = load_diffusion_checkpoint(args.checkpoint, map_location=args.device)
    training_config = loaded.payload.get("training_config", {})
    config = DiffusionTrainConfig(
        prepared_cache=str(args.prepared_cache),
        device=args.device,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        holdout_count=args.holdout_count,
        prediction_type=str(
            loaded.payload.get("model_config", {}).get(
                "prediction_type", loaded.model.prediction_type
            )
        ),
        architecture_variant=str(
            loaded.payload.get("model_config", {}).get(
                "architecture_variant", "v47"
            )
        ),
        target_parameterization=str(
            training_config.get(
                "target_parameterization",
                loaded.payload.get("model_config", {}).get(
                    "target_parameterization", loaded.model.prediction_type
                ),
            )
        ),
        target_steps=int(
            training_config.get(
                "target_steps",
                loaded.payload.get("model_config", {}).get("target_steps", 20),
            )
        ),
    )
    slices = load_apo2mol_slices(config)
    by_id = {value.sample_id: value for value in slices["test"].values}
    sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"sample ids not found in test split: {missing}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    loaded.model.eval()
    for index, sample_id in enumerate(sample_ids):
        value = _move_complex_to_device(by_id[sample_id], torch.device(args.device))
        state = sample_diffusion_state(
            value,
            1.0,
            translation_noise_scale=0.0,
            rotation_noise_scale=0.0,
            chi_noise_scale=0.0,
            target_parameterization=config.target_parameterization,
            target_steps=config.target_steps,
        )
        trajectory = sample_reverse_trajectory(
            loaded.model,
            state,
            steps=args.steps,
            motion_parameterization=str(
                training_config.get("motion_parameterization", "remaining")
            ),
            prediction_type=loaded.model.prediction_type,
            use_chi=bool(
                loaded.payload.get("sampler_config", {}).get(
                    "use_chi",
                    training_config.get("use_chi", True),
                )
            ),
        )
        output = output_dir / (
            f"{index:02d}_{sample_id.replace('.', '-')}_20step_trajectory.pdb"
        )
        _write_multimodel_pdb(output, value, trajectory.states)
        summary.append(
            {
                "sample_id": sample_id,
                "path": str(output),
                "model_count": len(trajectory.states),
                "steps": args.steps,
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "steps": args.steps,
                "samples": summary,
                "note": "Each PDB contains apo plus 20 predicted states; ligand is fixed as HETATM reference.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
