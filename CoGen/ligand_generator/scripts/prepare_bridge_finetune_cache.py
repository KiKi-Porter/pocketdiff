"""Build TargetDiff-compatible bridge-state samples from frozen PocketDiff.

The builder uses only apo coordinates, protein features, and the reference
ligand condition while constructing PocketDiff states. Holo protein
coordinates are never passed to the PocketDiff model. The saved TargetDiff
sample contains the generated protein state and the real ligand as diffusion
target x0.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
POCKET_ROOT = ROOT.parent / "pocketdiff"
import sys
sys.path.insert(0, str(POCKET_ROOT))

from pocketdiff.diffusion import apply_local_se3_update, build_diffusion_state_from_current
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.preprocessing.diffusion_cache import load_diffusion_cache
from pocketdiff.training.diffusion import load_diffusion_checkpoint


TYPE_TO_ELEMENT = torch.tensor([1, 6, 6, 7, 7, 8, 8, 9, 15, 15, 16, 16, 17], dtype=torch.long)


def _move(value, device):
    fields = {}
    for field in value.__dataclass_fields__.values():
        item = getattr(value, field.name)
        fields[field.name] = item.to(device) if isinstance(item, torch.Tensor) else item
    return type(value)(**fields)


def _pocket_state(value, protein_pos, ligand_pos, ligand_type, time_value):
    condition = replace(value, ligand_pos_ref=ligand_pos, ligand_type_ref=ligand_type)
    return build_diffusion_state_from_current(
        condition,
        protein_pos,
        float(time_value),
        ligand_pos=ligand_pos,
        ligand_type=ligand_type,
        ligand_condition_t=max(0, int(round(199.0 * float(time_value)))),
        target_parameterization="bridge_velocity",
        target_steps=20,
    )


@torch.no_grad()
def _update(pocket_model, value, protein_pos, ligand_pos, ligand_type, state_index):
    time_value = 1.0 - float(state_index) / 20.0
    state = _pocket_state(value, protein_pos, ligand_pos, ligand_type, time_value)
    prediction = pocket_model(state.model_input)
    frames = build_residue_frames(
        protein_pos,
        state.model_input.atom_to_residue,
        state.model_input.protein_atom_name,
        num_residues=state.model_input.residue_type.shape[0],
        atom_indices=state.model_input.frame_atom_indices,
    )
    valid = state.model_input.frame_valid & frames.valid
    return apply_local_se3_update(
        protein_pos,
        state.model_input.atom_to_residue,
        frames.origins,
        frames.frames,
        prediction.translation_local,
        prediction.rotation_local,
        fraction=1.0 / 20.0,
        frame_valid=valid,
    )


def _sample_dict(value):
    ligand_type = value.ligand_type_ref.cpu().long()
    ligand_pos = value.ligand_pos_ref.cpu().float()
    protein_element = value.protein_element.cpu().long()
    return {
        "protein_element": protein_element,
        "ligand_element": TYPE_TO_ELEMENT[ligand_type],
        "ligand_pos": ligand_pos,
        "protein_atom_feature": value.protein_feature.cpu().float(),
        "ligand_atom_feature_full": ligand_type,
    }


def _select(values, count):
    if count >= len(values):
        return list(values)
    indices = torch.linspace(0, len(values) - 1, count).round().long().tolist()
    return [values[i] for i in indices]


def build(args):
    device = torch.device(args.device)
    cache = load_diffusion_cache(args.pocket_cache)
    loaded = load_diffusion_checkpoint(args.pocket_checkpoint, map_location=str(device))
    pocket_model = loaded.model.eval()
    if pocket_model.prediction_type != "bridge_velocity":
        raise ValueError("PocketDiff checkpoint must use bridge_velocity")
    per_split = {
        "train": args.train_complexes,
        "valid": args.valid_complexes,
        "test": args.test_complexes,
    }
    complexes = []
    split_indices = {"train": [], "valid": [], "test": []}
    records = []
    for split in ("train", "valid", "test"):
        selected = _select(cache["splits"][split], per_split[split])
        for value in selected:
            value = _move(value, device)
            ligand_pos = value.ligand_pos_ref.float()
            ligand_type = value.ligand_type_ref.long()
            protein_pos = value.protein_pos_apo.float().clone()
            protein_states = []
            for state_index in range(args.states_per_complex):
                protein_pos = _update(
                    pocket_model, value, protein_pos, ligand_pos, ligand_type,
                    min(state_index, 19),
                )
                protein_states.append(protein_pos.detach().cpu().float())
            complex_index = len(complexes)
            split_indices[split].append(complex_index)
            complexes.append({
                "sample_id": value.sample_id,
                "static": _sample_dict(value),
                "protein_pos_states": torch.stack(protein_states),
                "split": split,
            })
            records.append({"sample_id": value.sample_id, "split": split, "states": args.states_per_complex})
    payload = {
        "metadata": {
            "cache_version": "targetdiff_bridge_finetune_v1",
            "source_cache": str(args.pocket_cache.resolve()),
            "pocketdiff_checkpoint": str(args.pocket_checkpoint.resolve()),
            "feature_contract": {"protein_atom_feature_dim": 27, "ligand_atom_class_dim": 13},
            "holo_protein_in_model_input": False,
            "states_per_complex": args.states_per_complex,
            "complex_counts": per_split,
        },
        "complexes": complexes,
        "records": records,
        "complex_indices_by_split": split_indices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "output": str(args.output.resolve()),
        "complex_count": len(complexes),
        "state_count": sum(int(item["protein_pos_states"].shape[0]) for item in complexes),
        "split_complex_counts": {key: len(value) for key, value in split_indices.items()},
        "complex_counts": per_split,
        "states_per_complex": args.states_per_complex,
        "finite": all(torch.isfinite(item["protein_pos_states"]).all() for item in complexes),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-cache", type=Path, default=POCKET_ROOT / "data/prepared_diffusion_full_v47.pt")
    parser.add_argument("--pocket-checkpoint", type=Path, default=POCKET_ROOT / "checkpoints/pocketdiff_v48_base.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/bridge_finetune_3000.pt")
    parser.add_argument("--report", type=Path, default=ROOT / "data/bridge_finetune_3000_report.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-complexes", type=int, default=2400)
    parser.add_argument("--valid-complexes", type=int, default=300)
    parser.add_argument("--test-complexes", type=int, default=300)
    parser.add_argument("--states-per-complex", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.train_complexes, args.valid_complexes, args.test_complexes = 2, 1, 1
        args.states_per_complex = 2
        args.output = ROOT / "data/bridge_finetune_smoke.pt"
        args.report = ROOT / "data/bridge_finetune_smoke_report.json"
    build(args)


if __name__ == "__main__":
    main()
