import torch
import pytest

from pocketdiff.diffusion.model import diffusion_motion_loss
from pocketdiff.diffusion.state import DiffusionStateInput, DiffusionStateTarget
from pocketdiff.geometry.so3 import so3_exp
from pocketdiff.models.directional_features import (
    LigandProteinVectorCrossMessage,
    build_residue_local_directional_features,
)
from pocketdiff.diffusion.model import DiffusionMotionPrediction


def _synthetic_geometry():
    # N, CA, C, CB for two residues.
    protein_pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.5],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 0.0, 1.5],
        ],
        dtype=torch.float32,
    )
    names = ("N", "CA", "C", "CB", "N", "CA", "C", "CB")
    atom_to_residue = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    frame_valid = torch.tensor([True, True])
    ligand_pos = torch.tensor(
        [[0.5, 0.5, 2.0], [3.4, 0.5, 2.0]],
        dtype=torch.float32,
    )
    return protein_pos, names, atom_to_residue, frame_valid, ligand_pos


def test_local_directional_features_are_global_rigid_invariant():
    protein_pos, names, atom_to_residue, frame_valid, ligand_pos = _synthetic_geometry()
    base = build_residue_local_directional_features(
        protein_pos, atom_to_residue, names, frame_valid, ligand_pos
    )
    rotation = so3_exp(torch.tensor([0.2, -0.1, 0.3]))
    shift = torch.tensor([5.0, -2.0, 1.0])
    moved = build_residue_local_directional_features(
        protein_pos @ rotation.transpose(0, 1) + shift,
        atom_to_residue,
        names,
        frame_valid,
        ligand_pos @ rotation.transpose(0, 1) + shift,
    )
    assert base.shape == (2, 13)
    assert torch.allclose(base, moved, atol=2.0e-5)


def test_local_directional_features_change_with_ligand_direction():
    protein_pos, names, atom_to_residue, frame_valid, ligand_pos = _synthetic_geometry()
    base = build_residue_local_directional_features(
        protein_pos, atom_to_residue, names, frame_valid, ligand_pos
    )
    moved_ligand = ligand_pos + torch.tensor([0.0, 0.0, 2.0])
    changed = build_residue_local_directional_features(
        protein_pos, atom_to_residue, names, frame_valid, moved_ligand
    )
    assert not torch.allclose(base, changed)


def test_vector_cross_message_is_global_rigid_invariant_and_differentiable():
    protein_pos, names, atom_to_residue, frame_valid, ligand_pos = _synthetic_geometry()
    from pocketdiff.geometry.frames import build_residue_frames

    frames = build_residue_frames(
        protein_pos,
        atom_to_residue,
        names,
        num_residues=2,
    )
    residue_hidden = torch.randn(2, 128, requires_grad=True)
    ligand_hidden = torch.randn(2, 128, requires_grad=True)
    message = LigandProteinVectorCrossMessage(hidden_dim=128)
    base = message(
        residue_hidden,
        frames.origins,
        frames.frames,
        frame_valid,
        ligand_hidden,
        ligand_pos,
    )
    rotation = so3_exp(torch.tensor([0.2, -0.1, 0.3]))
    shift = torch.tensor([5.0, -2.0, 1.0])
    moved_pos = protein_pos @ rotation.transpose(0, 1) + shift
    moved_ligand = ligand_pos @ rotation.transpose(0, 1) + shift
    moved_frames = build_residue_frames(
        moved_pos,
        atom_to_residue,
        names,
        num_residues=2,
    )
    moved = message(
        residue_hidden,
        moved_frames.origins,
        moved_frames.frames,
        frame_valid,
        ligand_hidden,
        moved_ligand,
    )
    assert torch.allclose(base, moved, atol=2.0e-5)
    base.sum().backward()
    assert residue_hidden.grad is not None
    assert ligand_hidden.grad is not None
    assert torch.isfinite(residue_hidden.grad).all()
    assert torch.isfinite(ligand_hidden.grad).all()


def test_score_normalized_loss_uses_bounded_time_scale():
    model_input = DiffusionStateInput(
        protein_pos=torch.zeros(1, 3),
        protein_feature=torch.zeros(1, 27),
        atom_to_residue=torch.zeros(1, dtype=torch.long),
        residue_type=torch.zeros(1, dtype=torch.long),
        frame_valid=torch.ones(1, dtype=torch.bool),
        chi_current=torch.zeros(1, 5),
        chi_mask=torch.zeros(1, 5, dtype=torch.bool),
        ligand_pos=torch.zeros(1, 3),
        ligand_type=torch.zeros(1, dtype=torch.long),
        diffusion_time=torch.tensor([0.5]),
        protein_atom_name=("CA",),
        protein_residue_name=("ALA",),
    )
    target = DiffusionStateTarget(
        translation_target_local=torch.zeros(1, 3),
        rotation_target_local=torch.zeros(1, 3),
        chi_target=torch.zeros(1, 5),
        target_valid=torch.ones(1, dtype=torch.bool),
        protein_pos_holo=torch.zeros(1, 3),
    )
    prediction = DiffusionMotionPrediction(
        translation_local=torch.ones(1, 3),
        rotation_local=torch.zeros(1, 3),
        chi=torch.zeros(1, 5),
    )
    plain = diffusion_motion_loss(prediction, target)
    normalized = diffusion_motion_loss(
        prediction,
        target,
        model_input=model_input,
        score_normalize=True,
    )
    assert torch.isfinite(normalized.loss)
    assert float(normalized.translation_loss) > float(plain.translation_loss)


def test_bridge_rate_loss_divides_remaining_target_by_current_time():
    model_input = DiffusionStateInput(
        protein_pos=torch.zeros(1, 3),
        protein_feature=torch.zeros(1, 27),
        atom_to_residue=torch.zeros(1, dtype=torch.long),
        residue_type=torch.zeros(1, dtype=torch.long),
        frame_valid=torch.ones(1, dtype=torch.bool),
        chi_current=torch.zeros(1, 5),
        chi_mask=torch.zeros(1, 5, dtype=torch.bool),
        ligand_pos=torch.zeros(1, 3),
        ligand_type=torch.zeros(1, dtype=torch.long),
        diffusion_time=torch.tensor([0.5]),
        protein_atom_name=("CA",),
        protein_residue_name=("ALA",),
    )
    target = DiffusionStateTarget(
        translation_target_local=torch.full((1, 3), 0.5),
        rotation_target_local=torch.zeros(1, 3),
        chi_target=torch.zeros(1, 5),
        target_valid=torch.ones(1, dtype=torch.bool),
        protein_pos_holo=torch.zeros(1, 3),
    )
    rate_prediction = DiffusionMotionPrediction(
        translation_local=torch.ones(1, 3),
        rotation_local=torch.zeros(1, 3),
        chi=torch.zeros(1, 5),
    )
    objective = diffusion_motion_loss(
        rate_prediction,
        target,
        model_input=model_input,
        motion_parameterization="bridge_rate",
    )
    assert float(objective.translation_loss) <= 1.0e-12
    with pytest.raises(ValueError, match="motion_parameterization"):
        diffusion_motion_loss(
            rate_prediction,
            target,
            model_input=model_input,
            motion_parameterization="bad",
        )
