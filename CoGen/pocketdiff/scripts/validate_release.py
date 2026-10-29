"""Validate the clean v4.8 release without using the original worktree."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from pocketdiff.diffusion import apply_local_se3_update, build_diffusion_state_from_current
from pocketdiff.diffusion.model import sample_reverse_trajectory
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.preprocessing.diffusion_cache import load_diffusion_cache
from pocketdiff.training.diffusion import (
    DiffusionTrainConfig,
    load_diffusion_checkpoint,
    train_diffusion_model,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "prepared_diffusion_full_v47.pt"
CHECKPOINT = ROOT / "checkpoints" / "pocketdiff_v48_base.pt"
REPORT = ROOT / "validation_report.json"


def _move_complex(value, device):
    fields = {}
    for field in value.__dataclass_fields__.values():
        item = getattr(value, field.name)
        fields[field.name] = item.to(device) if isinstance(item, torch.Tensor) else item
    return type(value)(**fields)


def main() -> int:
    device = torch.device("cpu")
    cache = load_diffusion_cache(CACHE)
    value = _move_complex(cache["splits"]["test"][0], device)
    loaded = load_diffusion_checkpoint(CHECKPOINT, map_location="cpu")
    model = loaded.model
    model.eval()
    state = build_diffusion_state_from_current(
        value,
        value.protein_pos_apo,
        1.0,
        ligand_pos=value.ligand_pos_ref,
        ligand_type=value.ligand_type_ref,
        target_parameterization="bridge_velocity",
        target_steps=20,
    )
    with torch.no_grad():
        prediction = model(state.model_input)
    frames = build_residue_frames(
        state.model_input.protein_pos,
        state.model_input.atom_to_residue,
        state.model_input.protein_atom_name,
        num_residues=value.num_residues,
        atom_indices=state.model_input.frame_atom_indices,
    )
    valid = state.model_input.frame_valid & frames.valid
    updated = apply_local_se3_update(
        state.model_input.protein_pos,
        state.model_input.atom_to_residue,
        frames.origins,
        frames.frames,
        prediction.translation_local,
        prediction.rotation_local,
        fraction=1.0 / 20.0,
        frame_valid=valid,
    )
    trajectory = sample_reverse_trajectory(
        model,
        state,
        steps=1,
        prediction_type="bridge_velocity",
        use_chi=False,
    )
    config = DiffusionTrainConfig(
        prepared_cache=str(CACHE),
        device="cpu",
        train_count=1,
        valid_count=0,
        test_count=0,
        holdout_count=0,
        updates=1,
        batch_size=1,
        encoder_backend="targetdiff",
        architecture_variant="v47",
        prediction_type="bridge_velocity",
        target_parameterization="bridge_velocity",
        target_steps=20,
        backbone_only=True,
        use_chi=False,
        endpoint_weight=0.0,
        backbone_endpoint_weight=0.0,
        next_state_weight=0.0,
        continuity_weight=0.0,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
    )
    _, _, rows = train_diffusion_model([value], config)
    report = {
        "release_root": str(ROOT),
        "cache": str(CACHE),
        "checkpoint": str(CHECKPOINT),
        "cache_counts": {key: len(items) for key, items in cache["splits"].items()},
        "cache_hash": cache["cache_hash"],
        "checkpoint_format": loaded.payload.get("format"),
        "encoder_backend": model.encoder_backend,
        "architecture_variant": model.architecture_variant,
        "prediction_type": model.prediction_type,
        "model_forward_finite": bool(
            torch.isfinite(prediction.translation_local).all()
            and torch.isfinite(prediction.rotation_local).all()
        ),
        "se3_update_finite": bool(torch.isfinite(updated).all()),
        "se3_update_changed_coordinates": bool(not torch.equal(updated, state.model_input.protein_pos)),
        "one_step_trajectory_states": len(trajectory.states),
        "one_update_training_rows": len(rows),
        "one_update_loss_finite": bool(rows and torch.isfinite(torch.tensor(float(rows[0]["loss"])))),
    }
    report["passed"] = bool(
        report["cache_counts"]["train"] == 20195
        and report["cache_counts"]["valid"] == 985
        and report["cache_counts"]["test"] == 450
        and report["checkpoint_format"] == "pocketdiff-diffusion-v48-directed-backbone"
        and report["model_forward_finite"]
        and report["se3_update_finite"]
        and report["se3_update_changed_coordinates"]
        and report["one_step_trajectory_states"] == 2
        and report["one_update_training_rows"] == 1
        and report["one_update_loss_finite"]
    )
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
