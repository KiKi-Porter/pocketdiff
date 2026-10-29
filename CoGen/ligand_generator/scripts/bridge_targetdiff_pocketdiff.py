"""Run matched TargetDiff-only and TargetDiff/PocketDiff bridge experiments.

All conditions use identical initial ligand noise and identical reverse
diffusion randomness:

1. original TargetDiff with a fixed apo protein;
2. bridge-adapted TargetDiff with a fixed apo protein;
3. that same TargetDiff with the source PocketDiff checkpoint;
4. that same TargetDiff with a noisy-ligand-finetuned PocketDiff checkpoint.

Holo coordinates are used only for diagnostics and are never passed to either
model. Each trajectory is written as one multi-model PDB file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
POCKET_ROOT = ROOT.parent / "pocketdiff"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(POCKET_ROOT))

from easydict import EasyDict
from pocketdiff.data.schema import PocketComplex
from pocketdiff.diffusion import apply_local_se3_update, build_diffusion_state_from_current
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.training.diffusion import load_diffusion_checkpoint
from pocketdiff.preprocessing.diffusion_cache import load_diffusion_cache
from models.molopt_score_model import ScorePosNet3D, index_to_log_onehot, log_sample_categorical
from utils import misc


DEFAULT_POCKET_CACHE = POCKET_ROOT / "data/prepared_diffusion_full_v47.pt"
DEFAULT_POCKET_CHECKPOINT = POCKET_ROOT / "runs/v48_full_dataset_epoch_equiv_20260930/checkpoint_best.pt"
DEFAULT_ORIGINAL_CHECKPOINT = ROOT / "pretrained_models/pretrained_diffusion.pt"
DEFAULT_INTERNAL_CHECKPOINT = ROOT / "runs/bridge_finetune_full_20261005/internal_adapter_best.pt"
DEFAULT_POCKET_ADAPTER_CHECKPOINT = POCKET_ROOT / "checkpoints/pocketdiff_noisy_ligand_ft.pt"
TARGETDIFF_LIGAND_ELEMENTS = (
    "H", "C", "C", "N", "N", "O", "O", "F", "P", "P", "S", "S", "Cl"
)


def _load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = ScorePosNet3D(config.model, 27, 13).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, payload


def _move_value(value: PocketComplex, device: torch.device) -> PocketComplex:
    fields = {}
    for field in value.__dataclass_fields__.values():
        item = getattr(value, field.name)
        fields[field.name] = item.to(device) if isinstance(item, torch.Tensor) else item
    return type(value)(**fields)


def _center_value(value: PocketComplex):
    center = value.protein_pos_apo.mean(dim=0)
    return replace(
        value,
        protein_pos_apo=value.protein_pos_apo - center,
        protein_pos_holo=value.protein_pos_holo - center,
        ligand_pos_ref=value.ligand_pos_ref - center,
        center_offset=torch.zeros_like(value.center_offset),
    ), center


def _sample_uniform_classes(count: int, classes: int, generator, device):
    return torch.randint(classes, (count,), generator=generator, device=device)


def _make_trace(count: int, seed: int, device: torch.device):
    generator = torch.Generator(device=device.type).manual_seed(seed)
    return {
        t: {
            "position_noise": torch.randn((count, 3), generator=generator, device=device),
            "categorical_uniform": torch.rand((count, 13), generator=generator, device=device),
        }
        for t in range(1000)
    }


def _categorical_from_uniform(logits, uniform):
    # Same Gumbel-max sampler as TargetDiff, with explicit randomness for paired
    # condition comparisons.
    gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-30)) + 1e-30)
    return (gumbel + logits).argmax(dim=-1)


@torch.no_grad()
def _target_step(model, state, t: int, randomness):
    batch_protein = torch.zeros(state["protein_pos"].shape[0], dtype=torch.long, device=state["protein_pos"].device)
    batch_ligand = torch.zeros(state["ligand_pos"].shape[0], dtype=torch.long, device=state["protein_pos"].device)
    timestep = torch.tensor([t], dtype=torch.long, device=batch_protein.device)
    preds = model(
        protein_pos=state["protein_pos"], protein_v=state["protein_v"], batch_protein=batch_protein,
        init_ligand_pos=state["ligand_pos"], init_ligand_v=state["ligand_v"],
        batch_ligand=batch_ligand, time_step=timestep,
    )
    pos0 = preds["pred_ligand_pos"]
    v0 = preds["pred_ligand_v"]
    mean = model.q_pos_posterior(x0=pos0, xt=state["ligand_pos"], t=timestep, batch=batch_ligand)
    logvar = model.posterior_logvar[timestep][batch_ligand].unsqueeze(-1)
    nonzero = (1 - (timestep == 0).float())[batch_ligand].unsqueeze(-1)
    ligand_pos = mean + nonzero * (0.5 * logvar).exp() * randomness["position_noise"]
    log_v0 = torch.log_softmax(v0, dim=-1)
    log_vt = index_to_log_onehot(state["ligand_v"], model.num_classes)
    log_prob = model.q_v_posterior(log_v0, log_vt, timestep, batch_ligand)
    ligand_v = _categorical_from_uniform(log_prob, randomness["categorical_uniform"])
    return {**state, "ligand_pos": ligand_pos, "ligand_v": ligand_v}


def _pocket_state(value, protein_pos, ligand_pos, ligand_v, t: float, targetdiff_t: int):
    # The shared state builder also creates supervision targets. Give it an
    # apo-only view so target coordinates and cached apo/holo intersection
    # masks cannot influence inference features.
    apo_frames = build_residue_frames(
        value.protein_pos_apo,
        value.atom_to_residue,
        value.protein_atom_name,
        num_residues=value.num_residues,
        atom_indices=value.frame_atom_indices,
    )
    bridge_value = replace(
        value,
        protein_pos_holo=value.protein_pos_apo,
        chi_holo=value.chi_apo,
        frame_valid=apo_frames.valid,
        chi_mask=torch.ones_like(value.chi_mask),
        ligand_pos_ref=ligand_pos,
        ligand_type_ref=ligand_v,
    )
    return build_diffusion_state_from_current(
        bridge_value, protein_pos, t,
        ligand_pos=ligand_pos, ligand_type=ligand_v,
        ligand_condition_t=int(targetdiff_t),
        target_parameterization="bridge_velocity", target_steps=20,
    )


def _assert_holo_invariant_state(value, protein_pos, ligand_pos, ligand_v):
    first = _pocket_state(value, protein_pos, ligand_pos, ligand_v, 0.5, 99).model_input
    poisoned = replace(
        value,
        protein_pos_holo=value.protein_pos_holo + 1000.0,
        chi_holo=value.chi_holo + 0.7,
        frame_valid=~value.frame_valid,
        chi_mask=~value.chi_mask,
    )
    second = _pocket_state(poisoned, protein_pos, ligand_pos, ligand_v, 0.5, 99).model_input
    for name in first.__dataclass_fields__:
        left, right = getattr(first, name), getattr(second, name)
        if isinstance(left, torch.Tensor):
            equal = isinstance(right, torch.Tensor) and torch.equal(left, right)
        else:
            equal = left == right
        if not equal:
            raise AssertionError(f"PocketDiff model input depends on holo-derived field: {name}")


@torch.no_grad()
def _pocket_update(pocket_model, value, state, pocket_index: int):
    t = 1.0 - pocket_index / 20.0
    targetdiff_t = 199 - 10 * pocket_index
    pocket = _pocket_state(
        value,
        state["protein_pos"],
        state["ligand_pos"],
        state["ligand_v"],
        t,
        targetdiff_t,
    )
    prediction = pocket_model(pocket.model_input)
    inp = pocket.model_input
    frames = build_residue_frames(
        state["protein_pos"], inp.atom_to_residue, inp.protein_atom_name,
        num_residues=inp.residue_type.shape[0], atom_indices=inp.frame_atom_indices,
    )
    valid = inp.frame_valid & frames.valid
    next_protein = apply_local_se3_update(
        state["protein_pos"], inp.atom_to_residue, frames.origins, frames.frames,
        prediction.translation_local, prediction.rotation_local,
        fraction=1.0 / 20.0, frame_valid=valid,
    )
    displacement = torch.linalg.vector_norm(next_protein - state["protein_pos"], dim=-1)
    return {**state, "protein_pos": next_protein}, {
        "targetdiff_t": targetdiff_t,
        "pocket_index": pocket_index,
        "valid_residues": int(valid.sum()),
        "mean_protein_displacement": float(displacement.mean()),
        "max_protein_displacement": float(displacement.max()),
        "changed": bool(not torch.equal(next_protein, state["protein_pos"])),
    }


@torch.no_grad()
def _run_condition(target_model, pocket_model, value, init_state, trace, coupled: bool):
    state = init_state
    protein_states = [state["protein_pos"].clone()]
    ligand_states = [(state["ligand_pos"].clone(), state["ligand_v"].clone())]
    pocket_diagnostics = []
    for t in range(999, -1, -1):
        if coupled and t <= 199 and t % 10 == 9:
            state, diagnostic = _pocket_update(
                pocket_model, value, state, (199 - t) // 10,
            )
            pocket_diagnostics.append(diagnostic)
        state = _target_step(target_model, state, t, trace[t])
        if t == 200 or (t <= 199 and t % 10 == 0) or t == 0:
            protein_states.append(state["protein_pos"].clone())
            ligand_states.append((state["ligand_pos"].clone(), state["ligand_v"].clone()))
    return state, protein_states, ligand_states, pocket_diagnostics


def _trajectory_is_finite(protein_states, ligand_states):
    return all(torch.isfinite(state).all() for state in protein_states) and all(
        torch.isfinite(position).all() and torch.isfinite(atom_type).all()
        for position, atom_type in ligand_states
    )


def _pdb_atom(record, serial, name, residue, chain, residue_id, position, element):
    try:
        residue_number = int(str(residue_id).strip())
    except ValueError:
        residue_number = 1
    atom_name = str(name).strip()[:4].rjust(4)
    return (f"{record:<6}{serial:5d} {atom_name} {residue[:3]:>3} {(chain or 'A')[:1]:1}{residue_number:4d}    "
            f"{float(position[0]):8.3f}{float(position[1]):8.3f}{float(position[2]):8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {element:>2}\n")


def _write_trajectory(path, value, protein_states, ligand_states, center):
    protein_elements = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "CL", 34: "SE"}
    residue_chain = [value.residue_chain_id[int(i)] for i in value.atom_to_residue.tolist()]
    residue_id = [value.residue_sequence_id[int(i)] for i in value.atom_to_residue.tolist()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for model_index, (protein, ligand) in enumerate(zip(protein_states, ligand_states), 1):
            handle.write(f"MODEL     {model_index:4d}\n")
            for serial, row in enumerate(zip(value.protein_atom_name, value.protein_residue_name,
                                             residue_chain, residue_id, value.protein_element.tolist(),
                                             (protein.cpu() + center.cpu()).tolist()), 1):
                name, residue, chain, rid, element, pos = row
                handle.write(_pdb_atom("ATOM", serial, name, residue, chain, rid, pos,
                                       protein_elements.get(int(element), "C")))
            ligand_pos, ligand_v = ligand
            ligand_rows = zip(
                ligand_v.cpu().tolist(),
                (ligand_pos.cpu() + center.cpu()).tolist(),
            )
            for offset, (atom_class, pos) in enumerate(ligand_rows, 1):
                element = TARGETDIFF_LIGAND_ELEMENTS[int(atom_class)]
                handle.write(_pdb_atom("HETATM", len(value.protein_atom_name) + offset,
                                       f"L{offset}", "LIG", "Z", "1", pos, element))
            handle.write("TER\nENDMDL\n")
        handle.write("END\n")


def _rmsd(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(torch.sqrt((a.sub(b).square().sum(dim=-1).mean()).clamp_min(0)).item())


def run(args):
    device = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = load_diffusion_cache(args.pocket_cache)
    test_values = cache["splits"]["test"][: args.samples]
    original, original_payload = _load_model(args.original_checkpoint, device)
    internal, internal_payload = _load_model(args.internal_checkpoint, device)
    pocket_loaded = load_diffusion_checkpoint(args.pocket_checkpoint, map_location=str(device))
    pocket_model = pocket_loaded.model.eval()
    if pocket_model.prediction_type != "bridge_velocity":
        raise ValueError(f"Expected bridge_velocity PocketDiff checkpoint, got {pocket_model.prediction_type}")
    adapter_pocket_model = None
    if args.pocket_adapter_checkpoint is not None:
        adapter_loaded = load_diffusion_checkpoint(
            args.pocket_adapter_checkpoint,
            map_location=str(device),
        )
        adapter_pocket_model = adapter_loaded.model.eval()
        if adapter_pocket_model.prediction_type != "bridge_velocity":
            raise ValueError(
                "Expected bridge_velocity PocketDiff adapter checkpoint, got "
                f"{adapter_pocket_model.prediction_type}"
            )
    records = []
    started = time.monotonic()
    for index, raw_value in enumerate(test_values):
        value, center = _center_value(raw_value)
        value = _move_value(value, device)
        _assert_holo_invariant_state(
            value,
            value.protein_pos_apo.float(),
            value.ligand_pos_ref.float(),
            value.ligand_type_ref.long(),
        )
        generator = torch.Generator(device=device.type).manual_seed(args.seed + index)
        protein_pos = value.protein_pos_apo.float()
        batch_protein = torch.zeros(protein_pos.shape[0], dtype=torch.long, device=device)
        ligand_count = value.ligand_pos_ref.shape[0]
        batch_ligand = torch.zeros(ligand_count, dtype=torch.long, device=device)
        init_ligand_pos = protein_pos.mean(dim=0, keepdim=True) + torch.randn((ligand_count, 3), generator=generator, device=device)
        init_ligand_v = _sample_uniform_classes(ligand_count, 13, generator, device)
        init = {"protein_pos": protein_pos, "protein_v": value.protein_feature.float(),
                "ligand_pos": init_ligand_pos, "ligand_v": init_ligand_v,
                "batch_protein": batch_protein, "batch_ligand": batch_ligand}
        trace = _make_trace(ligand_count, args.seed + index + 100000, device)
        original_final, original_protein, original_ligand, _ = _run_condition(original, pocket_model, value, init, trace, False)
        internal_final, internal_protein, internal_ligand, _ = _run_condition(internal, pocket_model, value, init, trace, False)
        bridge_final, bridge_protein, bridge_ligand, diagnostics = _run_condition(internal, pocket_model, value, init, trace, True)
        adapter_result = None
        if adapter_pocket_model is not None:
            adapter_result = _run_condition(
                internal, adapter_pocket_model, value, init, trace, True
            )
        trajectory_finite = all(
            _trajectory_is_finite(protein_states, ligand_states)
            for _, protein_states, ligand_states, _ in (
                (original_final, original_protein, original_ligand, []),
                (internal_final, internal_protein, internal_ligand, []),
                (bridge_final, bridge_protein, bridge_ligand, diagnostics),
            )
        )
        if adapter_result is not None:
            trajectory_finite = trajectory_finite and _trajectory_is_finite(
                adapter_result[1], adapter_result[2]
            )
        sample_dir = out / "samples" / f"{index:02d}_{value.sample_id.replace('.', '-') }"
        _write_trajectory(sample_dir / "original_targetdiff_trajectory.pdb", value, original_protein, original_ligand, center)
        _write_trajectory(sample_dir / "internal_cross_trajectory.pdb", value, internal_protein, internal_ligand, center)
        _write_trajectory(sample_dir / "bridged_trajectory.pdb", value, bridge_protein, bridge_ligand, center)
        if adapter_result is not None:
            _, adapter_protein, adapter_ligand, _ = adapter_result
            _write_trajectory(
                sample_dir / "noisy_ligand_pocket_bridge_trajectory.pdb",
                value,
                adapter_protein,
                adapter_ligand,
                center,
            )
        backbone = torch.tensor([n in {"N", "CA", "C", "O"} for n in value.protein_atom_name], device=device)
        record = {
            "sample_id": value.sample_id,
            "original_ligand_atoms": int(original_final["ligand_pos"].shape[0]),
            "internal_ligand_atoms": int(internal_final["ligand_pos"].shape[0]),
            "bridged_ligand_atoms": int(bridge_final["ligand_pos"].shape[0]),
            "apo_backbone_rmsd": _rmsd(value.protein_pos_apo, value.protein_pos_holo, backbone),
            "original_backbone_rmsd": _rmsd(original_final["protein_pos"], value.protein_pos_holo, backbone),
            "internal_backbone_rmsd": _rmsd(internal_final["protein_pos"], value.protein_pos_holo, backbone),
            "bridged_backbone_rmsd": _rmsd(bridge_final["protein_pos"], value.protein_pos_holo, backbone),
            "bridged_protein_displacement_mean": float(torch.linalg.vector_norm(bridge_final["protein_pos"] - value.protein_pos_apo, dim=-1).mean()),
            "bridged_protein_displacement_max": float(torch.linalg.vector_norm(bridge_final["protein_pos"] - value.protein_pos_apo, dim=-1).max()),
            "pocket_calls": len(diagnostics),
            "pocket_changed_calls": sum(int(item["changed"]) for item in diagnostics),
            "holo_invariance_passed": True,
            "all_trajectory_states_finite": bool(trajectory_finite),
            "pocket_diagnostics": diagnostics,
            "trajectories": {name: str((sample_dir / f"{name}_trajectory.pdb").resolve()) for name in ("original_targetdiff", "internal_cross", "bridged")},
        }
        if adapter_result is not None:
            adapter_final, _, _, adapter_diagnostics = adapter_result
            adapter_displacement = torch.linalg.vector_norm(
                adapter_final["protein_pos"] - value.protein_pos_apo,
                dim=-1,
            )
            record.update(
                {
                    "noisy_ligand_pocket_bridge_ligand_atoms": int(
                        adapter_final["ligand_pos"].shape[0]
                    ),
                    "noisy_ligand_pocket_bridge_backbone_rmsd": _rmsd(
                        adapter_final["protein_pos"],
                        value.protein_pos_holo,
                        backbone,
                    ),
                    "noisy_ligand_pocket_bridge_protein_displacement_mean": float(
                        adapter_displacement.mean()
                    ),
                    "noisy_ligand_pocket_bridge_protein_displacement_max": float(
                        adapter_displacement.max()
                    ),
                    "noisy_ligand_pocket_bridge_calls": len(adapter_diagnostics),
                    "noisy_ligand_pocket_bridge_changed_calls": sum(
                        int(item["changed"]) for item in adapter_diagnostics
                    ),
                    "noisy_ligand_pocket_bridge_diagnostics": adapter_diagnostics,
                    "noisy_ligand_pocket_bridge_finite": bool(
                        torch.isfinite(adapter_final["protein_pos"]).all()
                        and torch.isfinite(adapter_final["ligand_pos"]).all()
                    ),
                }
            )
            record["trajectories"]["noisy_ligand_pocket_bridge"] = str(
                (
                    sample_dir / "noisy_ligand_pocket_bridge_trajectory.pdb"
                ).resolve()
            )
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "version": "clean-bridge-phase1-20261004",
        "samples": len(records),
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        "original_checkpoint": str(args.original_checkpoint.resolve()),
        "internal_checkpoint": str(args.internal_checkpoint.resolve()),
        "pocket_checkpoint": str(args.pocket_checkpoint.resolve()),
        "pocket_adapter_checkpoint": (
            str(args.pocket_adapter_checkpoint.resolve())
            if args.pocket_adapter_checkpoint is not None
            else None
        ),
        "late_protocol": "prelude 999..200; 20 PocketDiff calls before TargetDiff blocks 199..190 through 9..0",
        "holo_is_model_input": False,
        "holo_used_for_pocket_state_construction": False,
        "holo_invariance_passed": bool(records and all(r["holo_invariance_passed"] for r in records)),
        "records": records,
        "all_finite": bool(
            records
            and all(
                r["holo_invariance_passed"]
                and r["all_trajectory_states_finite"]
                and r["pocket_calls"] == 20
                and (
                    "noisy_ligand_pocket_bridge_calls" not in r
                    or (
                        r["noisy_ligand_pocket_bridge_calls"] == 20
                        and r["noisy_ligand_pocket_bridge_finite"]
                    )
                )
                for r in records
            )
        ),
        "all_coupled_calls_changed_protein": bool(
            records
            and all(
                r["pocket_changed_calls"] == 20
                and (
                    "noisy_ligand_pocket_bridge_changed_calls" not in r
                    or r["noisy_ligand_pocket_bridge_changed_calls"] == 20
                )
                for r in records
            )
        ),
    }
    (out / ("smoke_report.json" if args.smoke else "comparison.json")).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-cache", type=Path, default=DEFAULT_POCKET_CACHE)
    parser.add_argument("--pocket-checkpoint", type=Path, default=DEFAULT_POCKET_CHECKPOINT)
    parser.add_argument(
        "--pocket-adapter-checkpoint",
        type=Path,
        default=DEFAULT_POCKET_ADAPTER_CHECKPOINT,
    )
    parser.add_argument("--original-checkpoint", type=Path, default=DEFAULT_ORIGINAL_CHECKPOINT)
    parser.add_argument("--internal-checkpoint", type=Path, default=DEFAULT_INTERNAL_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/bridge_phase1_clean_20261004")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.smoke == args.full:
        parser.error("choose exactly one of --smoke or --full")
    if args.smoke:
        args.samples = 1
    report = run(args)
    if not report["all_finite"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
