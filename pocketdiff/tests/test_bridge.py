import math

import torch

from pocketdiff.geometry.bridge import (
    apply_fractional_update,
    build_bridge_state,
    oracle_reconstruction_metrics,
    remaining_transform_current_to_holo,
)
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.geometry.so3 import so3_exp


def _rigid_pair():
    apo = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # N
            [0.0, 0.0, 0.0],  # CA
            [1.0, 0.0, 0.0],  # C
            [1.0, 1.0, 0.0],  # side atom
        ],
        dtype=torch.float32,
    )
    names = ["N", "CA", "C", "CB"]
    atom_to_residue = torch.zeros(4, dtype=torch.long)
    apo_frame = build_residue_frames(apo, atom_to_residue, names, num_residues=1)
    rotation = so3_exp(torch.tensor([0.0, 0.0, math.pi / 2.0]))
    origin = torch.tensor([[1.0, 2.0, 3.0]])
    holo = (apo - apo_frame.origins[0]) @ rotation.transpose(0, 1) + origin[0]
    holo_frame = build_residue_frames(holo, atom_to_residue, names, num_residues=1)
    return apo, holo, names, atom_to_residue, apo_frame, holo_frame


def test_remaining_transform_and_full_update_reconstruct_known_rigid_motion():
    apo, holo, names, atom_to_residue, apo_frame, holo_frame = _rigid_pair()
    remaining = remaining_transform_current_to_holo(
        apo_frame.origins,
        apo_frame.frames,
        holo_frame.origins,
        holo_frame.frames,
        frame_valid=apo_frame.valid & holo_frame.valid,
    )
    assert torch.allclose(remaining.translation_local, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-5)
    assert torch.allclose(
        remaining.rotvec_local,
        torch.tensor([[0.0, 0.0, math.pi / 2.0]]),
        atol=2e-4,
    )
    updated = apply_fractional_update(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        remaining.translation_local,
        remaining.rotvec_local,
        remaining_steps=1,
        frame_valid=remaining.valid,
    )
    assert torch.allclose(updated, holo, atol=1e-4)


def test_fractional_updates_with_recomputed_frames_reach_holo():
    apo, holo, names, atom_to_residue, _, holo_frame = _rigid_pair()
    current = apo.clone()
    for step in range(4):
        current_frame = build_residue_frames(current, atom_to_residue, names, num_residues=1)
        remaining = remaining_transform_current_to_holo(
            current_frame.origins,
            current_frame.frames,
            holo_frame.origins,
            holo_frame.frames,
            frame_valid=current_frame.valid & holo_frame.valid,
        )
        current = apply_fractional_update(
            current,
            atom_to_residue,
            current_frame.origins,
            current_frame.frames,
            remaining.translation_local,
            remaining.rotvec_local,
            remaining_steps=4 - step,
            frame_valid=remaining.valid,
        )
    assert torch.allclose(current, holo, atol=2e-4)


def test_bridge_state_endpoints_and_oracle_metrics():
    apo, holo, names, atom_to_residue, apo_frame, holo_frame = _rigid_pair()
    valid = apo_frame.valid & holo_frame.valid
    state_zero = build_bridge_state(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        holo_frame.origins,
        holo_frame.frames,
        fraction=0.0,
        frame_valid=valid,
    )
    state_one = build_bridge_state(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        holo_frame.origins,
        holo_frame.frames,
        fraction=1.0,
        frame_valid=valid,
    )
    assert torch.equal(state_zero.protein_pos, apo)
    assert torch.allclose(state_one.protein_pos, holo, atol=2e-4)
    metrics = oracle_reconstruction_metrics(state_one.protein_pos, holo, atom_to_residue, names, valid)
    assert metrics.atom_rmsd < 1e-4
    assert metrics.backbone_rmsd < 1e-4
    assert metrics.frame_invalid_count == 0


def test_invalid_residue_coordinates_are_bitwise_unchanged():
    apo, holo, names, atom_to_residue, apo_frame, holo_frame = _rigid_pair()
    # Add a residue with no N–CA–C frame and a separate atom group.
    positions = torch.cat((apo, torch.tensor([[7.0, 7.0, 7.0]])), dim=0)
    atom_ids = torch.cat((atom_to_residue, torch.tensor([1], dtype=torch.long)))
    names_with_invalid = names + ["CB"]
    current_frame = build_residue_frames(positions, atom_ids, names_with_invalid, num_residues=2)
    holo_origins = torch.cat((holo_frame.origins, torch.zeros(1, 3)), dim=0)
    holo_frames = torch.cat((holo_frame.frames, torch.eye(3).unsqueeze(0)), dim=0)
    remaining = remaining_transform_current_to_holo(
        current_frame.origins,
        current_frame.frames,
        holo_origins,
        holo_frames,
        frame_valid=current_frame.valid & torch.tensor([True, False]),
    )
    updated = apply_fractional_update(
        positions,
        atom_ids,
        current_frame.origins,
        current_frame.frames,
        remaining.translation_local,
        remaining.rotvec_local,
        remaining_steps=1,
        frame_valid=remaining.valid,
    )
    assert torch.equal(updated[-1], positions[-1])


def test_zero_update_is_bitwise_unchanged():
    apo, _, names, atom_to_residue, apo_frame, _ = _rigid_pair()
    zero = torch.zeros(1, 3)
    updated = apply_fractional_update(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        zero,
        zero,
        remaining_steps=20,
    )
    assert torch.equal(updated, apo)


def test_fractional_update_accepts_one_remaining_step_count_per_residue():
    apo, _, names, atom_to_residue, apo_frame, _ = _rigid_pair()
    translation = torch.tensor([[0.4, 0.0, 0.0]])
    rotvec = torch.zeros(1, 3)
    scalar = apply_fractional_update(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        translation,
        rotvec,
        remaining_steps=4,
    )
    vector = apply_fractional_update(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        translation,
        rotvec,
        remaining_steps=torch.tensor([4]),
    )
    assert torch.equal(scalar, vector)


def test_fractional_update_has_finite_gradients():
    apo, _, names, atom_to_residue, apo_frame, _ = _rigid_pair()
    translation = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
    rotvec = torch.tensor([[0.1, 0.2, -0.15]], requires_grad=True)
    updated = apply_fractional_update(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        translation,
        rotvec,
        remaining_steps=4,
    )
    loss = updated[0, 0] + 0.3 * updated[2, 1]
    loss.backward()
    assert translation.grad is not None and rotvec.grad is not None
    assert torch.isfinite(translation.grad).all()
    assert torch.isfinite(rotvec.grad).all()


def test_global_se3_equivariance_of_remaining_and_update():
    apo, holo, names, atom_to_residue, apo_frame, holo_frame = _rigid_pair()
    global_rotation = so3_exp(torch.tensor([0.2, -0.1, 0.3]))
    global_translation = torch.tensor([5.0, -2.0, 1.0])
    apo_global = apo @ global_rotation.transpose(0, 1) + global_translation
    holo_global = holo @ global_rotation.transpose(0, 1) + global_translation
    apo_frame_global = build_residue_frames(apo_global, atom_to_residue, names, num_residues=1)
    holo_frame_global = build_residue_frames(holo_global, atom_to_residue, names, num_residues=1)
    local = remaining_transform_current_to_holo(
        apo_frame.origins,
        apo_frame.frames,
        holo_frame.origins,
        holo_frame.frames,
    )
    local_global = remaining_transform_current_to_holo(
        apo_frame_global.origins,
        apo_frame_global.frames,
        holo_frame_global.origins,
        holo_frame_global.frames,
    )
    assert torch.allclose(local.translation_local, local_global.translation_local, atol=1e-4)
    assert torch.allclose(local.rotvec_local, local_global.rotvec_local, atol=1e-4)
    updated = apply_fractional_update(
        apo,
        atom_to_residue,
        apo_frame.origins,
        apo_frame.frames,
        local.translation_local,
        local.rotvec_local,
        remaining_steps=1,
    )
    updated_global = apply_fractional_update(
        apo_global,
        atom_to_residue,
        apo_frame_global.origins,
        apo_frame_global.frames,
        local_global.translation_local,
        local_global.rotvec_local,
        remaining_steps=1,
    )
    assert torch.allclose(updated_global, updated @ global_rotation.transpose(0, 1) + global_translation, atol=2e-4)
