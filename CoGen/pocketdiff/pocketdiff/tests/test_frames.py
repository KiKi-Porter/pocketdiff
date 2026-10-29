import pytest
import torch

from pocketdiff.geometry.frames import build_residue_frames, frame_orthogonality_error


def _toy_coordinates():
    # Residue 0 is a right-handed N–CA–C frame.  Residue 1 is collinear and
    # must be masked without producing NaN/Inf placeholders.
    pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # N0
            [0.0, 0.0, 0.0],  # CA0
            [1.0, 0.0, 0.0],  # C0
            [1.0, 0.0, 1.0],  # N1
            [2.0, 0.0, 1.0],  # CA1
            [3.0, 0.0, 1.0],  # C1; N1/CA1/C1 collinear
        ],
        dtype=torch.float32,
    )
    return pos, torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long), ["N", "CA", "C"] * 2


def test_frame_axis_order_and_valid_mask():
    pos, atom_to_residue, names = _toy_coordinates()
    result = build_residue_frames(pos, atom_to_residue, names, num_residues=2)

    assert result.valid.tolist() == [True, False]
    assert result.atom_indices.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert torch.allclose(result.origins[0], torch.zeros(3))
    assert torch.allclose(result.frames[0], torch.eye(3), atol=1e-6)
    assert torch.allclose(result.frames[1], torch.eye(3), atol=1e-6)
    assert float(frame_orthogonality_error(result.frames)) < 1e-5
    assert torch.all(torch.isfinite(result.frames))


def test_missing_frame_atom_is_finite_and_invalid():
    pos = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    atom_to_residue = torch.zeros(2, dtype=torch.long)
    result = build_residue_frames(pos, atom_to_residue, ["N", "CA"], num_residues=1)
    assert not bool(result.valid[0])
    assert torch.equal(result.frames[0], torch.eye(3))
    assert torch.equal(result.origins[0], torch.zeros(3))


def test_frames_are_differentiable_for_valid_coordinates():
    pos, atom_to_residue, names = _toy_coordinates()
    pos = pos.clone().requires_grad_(True)
    result = build_residue_frames(pos, atom_to_residue, names, num_residues=2)
    # Use a non-constant projection; the Frobenius norm of an orthonormal
    # frame is intentionally constant and would yield a zero gradient.
    loss = result.origins[0].sum() + result.frames[0, 0, 1] + 0.1 * result.frames[0, 1, 2]
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert pos.grad.abs().sum() > 0


def test_invalid_arguments_are_rejected():
    pos, atom_to_residue, names = _toy_coordinates()
    with pytest.raises(ValueError, match="out-of-range"):
        build_residue_frames(pos, atom_to_residue + 3, names, num_residues=2)
