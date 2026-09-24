import torch

from pocketdiff_v4.geometry import (
    apply_chi_sparse,
    apply_motion,
    residue_frames,
    residue_level_names,
)


def _rotation():
    axis = torch.tensor([0.3, -0.7, 0.2])
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.83)
    x, y, z = axis
    skew = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float32
    )
    return (
        torch.eye(3)
        + torch.sin(angle) * skew
        + (1.0 - torch.cos(angle)) * (skew @ skew)
    )


def _protein():
    return torch.tensor(
        [
            [-0.4, 0.2, 0.1],  # N
            [0.0, 0.0, 0.0],  # CA
            [0.2, 0.8, -0.1],  # C
            [0.3, -0.5, 0.4],
            [0.8, -0.2, 0.7],
        ],
        dtype=torch.float32,
    )


def test_residue_frame_is_orthonormal_and_right_handed():
    pos = _protein()
    origin, frame, valid = residue_frames(pos, torch.tensor([[0, 1, 2]]))
    assert valid.tolist() == [True]
    assert torch.allclose(origin[0], pos[1])
    assert torch.allclose(frame[0].T @ frame[0], torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.det(frame), torch.ones(1), atol=1e-6)


def test_apply_motion_commutes_with_global_rigid_transform():
    pos = _protein()
    atom_to_residue = torch.zeros(pos.shape[0], dtype=torch.long)
    frame_index = torch.tensor([[0, 1, 2]])
    geometry_mask = torch.zeros((1, 5), dtype=torch.bool)
    sample = {
        "atom_to_residue": atom_to_residue,
        "frame_index": frame_index,
        "chi_geometry_mask": geometry_mask,
        "chi_axis": torch.full((1, 5, 2), -1, dtype=torch.long),
        "chi_ptr": torch.zeros(6, dtype=torch.long),
        "chi_downstream": torch.empty(0, dtype=torch.long),
    }
    translation = torch.tensor([[0.4, -0.1, 0.2]])
    rotation = torch.tensor([[0.2, -0.3, 0.1]])
    chi = torch.zeros((1, 5))
    moved = apply_motion(sample, pos, translation, rotation, chi)

    global_rotation = _rotation()
    global_translation = torch.tensor([2.0, -1.5, 0.7])
    transformed_pos = pos @ global_rotation.T + global_translation
    transformed_moved = apply_motion(
        sample, transformed_pos, translation, rotation, chi
    )
    expected = moved @ global_rotation.T + global_translation
    assert torch.allclose(transformed_moved, expected, atol=2e-6, rtol=2e-6)


def test_sparse_chi_update_is_rigid_transform_equivariant():
    pos = _protein()
    axis = torch.full((1, 5, 2), -1, dtype=torch.long)
    axis[0, 0] = torch.tensor([1, 3])
    ptr = torch.tensor([0, 1, 1, 1, 1, 1])
    downstream = torch.tensor([4])
    mask = torch.tensor([[True, False, False, False, False]])
    delta = torch.tensor([[0.7, 0.0, 0.0, 0.0, 0.0]])
    moved = apply_chi_sparse(pos, delta, axis, ptr, downstream, mask)

    q = _rotation()
    shift = torch.tensor([-0.8, 0.3, 1.1])
    transformed = pos @ q.T + shift
    transformed_moved = apply_chi_sparse(
        transformed, delta, axis, ptr, downstream, mask
    )
    assert torch.allclose(transformed_moved, moved @ q.T + shift, atol=2e-6)
    assert torch.allclose(moved[:4], pos[:4])


def test_residue_name_pooling_validates_per_atom_names():
    names = residue_level_names(
        ["N", "CA", "CB", "N", "CA"],
        ["SER", "SER", "SER", "GLY", "GLY"],
        torch.tensor([0, 0, 0, 1, 1]),
        2,
    )
    assert names == ["SER", "GLY"]
