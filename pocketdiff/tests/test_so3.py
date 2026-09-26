import math

import pytest
import torch

from pocketdiff.geometry.so3 import (
    so3_compose,
    so3_exp,
    so3_geodesic_angle,
    so3_inverse,
    so3_is_proper,
    so3_log,
)


def test_exp_identity_and_known_z_rotation():
    zero = torch.zeros(3)
    assert torch.equal(so3_exp(zero), torch.eye(3))
    angle = math.pi / 2.0
    matrix = so3_exp(torch.tensor([0.0, 0.0, angle]))
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(matrix, expected, atol=1e-6)
    assert bool(so3_is_proper(matrix))


def test_log_exp_round_trip_including_near_pi():
    vectors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.3, 0.4],
            [(math.pi - 1.0e-5), 0.0, 0.0],
            [0.0, -(math.pi - 1.0e-5), 0.0],
            [0.0, 0.0, math.pi],
            [0.4, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    matrices = so3_exp(vectors)
    recovered = so3_exp(so3_log(matrices))
    assert torch.allclose(recovered, matrices, atol=2e-4)
    assert torch.all(so3_log(matrices).isfinite())


def test_inverse_and_composition():
    first = so3_exp(torch.tensor([0.3, -0.1, 0.2]))
    second = so3_exp(torch.tensor([-0.2, 0.4, 0.1]))
    composed = so3_compose(first, second)
    assert torch.allclose(composed @ so3_inverse(second), first, atol=1e-5)
    identity = so3_compose(composed, so3_inverse(composed))
    assert torch.allclose(identity, torch.eye(3), atol=1e-5)
    assert float(so3_geodesic_angle(composed, composed)) < 1e-6


def test_exp_log_have_finite_gradients_away_from_pi():
    vector = torch.tensor([0.4, -0.2, 0.3], requires_grad=True)
    matrix = so3_exp(vector)
    recovered = so3_log(matrix)
    loss = recovered.square().sum() + matrix[0, 1]
    loss.backward()
    assert vector.grad is not None
    assert torch.isfinite(vector.grad).all()
    assert vector.grad.abs().sum() > 0


def test_nonfinite_and_wrong_shapes_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        so3_exp(torch.tensor([float("nan"), 0.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        so3_log(torch.eye(2))


def test_log_resolves_small_angles_despite_float32_trace_roundoff():
    # Late bridge states contain sub-milliradian rotations. Build reference
    # matrices in float64, then simulate the float32 geometry/model input.
    angles = torch.tensor([1e-5, 1e-4, 3e-4, 7e-4, 1e-3, 3e-3], dtype=torch.float64)
    matrices = torch.eye(3, dtype=torch.float64).repeat(len(angles), 1, 1)
    matrices[:, 0, 0] = angles.cos()
    matrices[:, 1, 1] = angles.cos()
    matrices[:, 0, 1] = -angles.sin()
    matrices[:, 1, 0] = angles.sin()
    expected = torch.zeros(len(angles), 3)
    expected[:, 2] = angles.float()
    recovered = so3_log(matrices.float())
    torch.testing.assert_close(recovered, expected, atol=1e-7, rtol=0)


def test_log_small_angle_is_robust_to_frame_composition_roundoff():
    angle = 7e-4
    c, s = math.cos(angle), math.sin(angle)
    matrix = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    # Float32 frame products can shift trace by a few ulps while their skew
    # part still resolves the small angle. This must not corrupt the label.
    matrix[0, 0] -= 2e-7
    result = so3_log(matrix)
    torch.testing.assert_close(result, torch.tensor([0., 0., angle]), atol=1e-7, rtol=0)
