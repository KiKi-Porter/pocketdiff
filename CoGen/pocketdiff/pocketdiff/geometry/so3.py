"""Stable SO(3) exponential/logarithm utilities.

Rotation matrices use the usual column-axis convention required by the frame
definition.  Coordinates remain row vectors elsewhere in PocketDiff, so a
global active rotation is applied to coordinates as ``x @ R.T``.
"""

from __future__ import annotations

import math

import torch


Tensor = torch.Tensor
_SMALL_ANGLE = 1.0e-5
_PI_BRANCH = 1.0e-4


def _require_rotvec(rotvec: Tensor) -> Tensor:
    if not isinstance(rotvec, torch.Tensor) or rotvec.ndim < 1 or rotvec.shape[-1] != 3:
        raise ValueError("rotvec must have shape [..., 3]")
    if not rotvec.is_floating_point():
        raise TypeError("rotvec must use a floating dtype")
    if not torch.isfinite(rotvec).all():
        raise ValueError("rotvec contains non-finite values")
    return rotvec.to(dtype=torch.float32)


def _require_rotation_matrix(matrix: Tensor) -> Tensor:
    if not isinstance(matrix, torch.Tensor) or matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must have shape [..., 3, 3]")
    if not matrix.is_floating_point():
        raise TypeError("rotation matrix must use a floating dtype")
    if not torch.isfinite(matrix).all():
        raise ValueError("rotation matrix contains non-finite values")
    return matrix.to(dtype=torch.float32)


def _skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def so3_exp(rotvec: Tensor) -> Tensor:
    """Map axis-angle vectors to proper rotation matrices.

    The small-angle coefficients use Taylor expansions, avoiding a singular
    divide at zero while preserving a differentiable path for training.
    """

    vector = _require_rotvec(rotvec)
    theta_sq = (vector * vector).sum(dim=-1, keepdim=True)
    # Do not evaluate sqrt(0) even in an unselected torch.where branch:
    # its infinite derivative produces 0*inf NaNs during backpropagation.
    # The selected small-angle Taylor branch still uses the exact theta_sq.
    theta = torch.sqrt(theta_sq.clamp_min(1.0e-12))
    theta_sq_safe = theta_sq.clamp_min(1.0e-12)
    theta_safe = theta.clamp_min(1.0e-6)

    a_trig = torch.sin(theta) / theta_safe
    b_trig = (1.0 - torch.cos(theta)) / theta_sq_safe
    a_series = 1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0
    b_series = 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0
    small = theta_sq <= (_SMALL_ANGLE * _SMALL_ANGLE)
    coefficient_a = torch.where(small, a_series, a_trig)
    coefficient_b = torch.where(small, b_series, b_trig)

    skew = _skew(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + coefficient_a[..., None] * skew + coefficient_b[..., None] * (skew @ skew)


def _near_pi_axis(matrix: Tensor) -> Tensor:
    """Extract a deterministic axis for rotations whose angle is near pi."""

    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    axis_abs = torch.sqrt(((diagonal + 1.0) * 0.5).clamp_min(0.0))
    a0, a1, a2 = axis_abs.unbind(dim=-1)
    r01, r02, r10 = matrix[..., 0, 1], matrix[..., 0, 2], matrix[..., 1, 0]
    r12, r20, r21 = matrix[..., 1, 2], matrix[..., 2, 0], matrix[..., 2, 1]
    d0 = (4.0 * a0).clamp_min(1.0e-6)
    d1 = (4.0 * a1).clamp_min(1.0e-6)
    d2 = (4.0 * a2).clamp_min(1.0e-6)
    axis0 = torch.stack((a0, (r01 + r10) / d0, (r02 + r20) / d0), dim=-1)
    axis1 = torch.stack(((r01 + r10) / d1, a1, (r12 + r21) / d1), dim=-1)
    axis2 = torch.stack(((r02 + r20) / d2, (r12 + r21) / d2, a2), dim=-1)
    pivot = axis_abs.argmax(dim=-1)
    axis = torch.where((pivot == 0)[..., None], axis0, torch.where((pivot == 1)[..., None], axis1, axis2))
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-6)

    # Axis and -axis represent the same pi rotation.  A fixed sign makes the
    # branch deterministic, which is important for reproducible labels.
    pivot_value = axis.gather(-1, pivot[..., None]).squeeze(-1)
    sign = torch.where(pivot_value < 0.0, -torch.ones_like(pivot_value), torch.ones_like(pivot_value))
    return axis * sign[..., None]


def so3_log(matrix: Tensor) -> Tensor:
    """Return principal axis-angle vectors with angles in ``[0, pi]``."""

    rotation = _require_rotation_matrix(matrix)
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = 0.5 * torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(vee, dim=-1, keepdim=True)
    # acos(trace) loses relative precision for float32 rotations near identity:
    # a few ulps in trace can dominate the remaining late-bridge angle. The
    # skew part retains sin(theta), so atan2 resolves these small rotations.
    theta = torch.atan2(sine.squeeze(-1), cosine)
    regular = (theta < (math.pi - _PI_BRANCH)).unsqueeze(-1)
    small = (theta < _SMALL_ANGLE).unsqueeze(-1)
    regular_vector = vee * theta.unsqueeze(-1) / sine.clamp_min(1.0e-6)
    small_vector = vee
    pi_vector = _near_pi_axis(rotation) * theta.unsqueeze(-1)
    result = torch.where(regular, regular_vector, pi_vector)
    result = torch.where(small, small_vector, result)
    return result


def so3_inverse(matrix: Tensor) -> Tensor:
    """Invert a rotation matrix by transpose."""

    rotation = _require_rotation_matrix(matrix)
    return rotation.transpose(-1, -2)


def so3_compose(left: Tensor, right: Tensor) -> Tensor:
    """Compose column-style rotations as ``left @ right``."""

    return _require_rotation_matrix(left) @ _require_rotation_matrix(right)


def so3_geodesic_angle(first: Tensor, second: Tensor) -> Tensor:
    """Return the principal angle of ``first.T @ second`` in radians."""

    first_rotation = _require_rotation_matrix(first)
    second_rotation = _require_rotation_matrix(second)
    return torch.linalg.vector_norm(so3_log(first_rotation.transpose(-1, -2) @ second_rotation), dim=-1)


def so3_is_proper(matrix: Tensor, *, atol: float = 1.0e-5) -> Tensor:
    """Boolean diagnostic for orthogonal matrices with determinant +1."""

    rotation = _require_rotation_matrix(matrix)
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    orthogonal = (rotation.transpose(-1, -2) @ rotation - identity).abs().amax(dim=(-2, -1)) <= atol
    determinant = torch.linalg.det(rotation)
    return orthogonal & ((determinant - 1.0).abs() <= atol)


__all__ = [
    "so3_compose",
    "so3_exp",
    "so3_geodesic_angle",
    "so3_inverse",
    "so3_is_proper",
    "so3_log",
]
