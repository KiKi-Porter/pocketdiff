"""N–CA–C residue frames under the PocketDiff row-vector convention."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch


Tensor = torch.Tensor
FRAME_ATOMS = ("N", "CA", "C")


@lru_cache(maxsize=4096)
def _cached_frame_atom_indices(
    atom_names: tuple[str, ...],
    atom_to_residue: tuple[int, ...],
    num_residues: int,
) -> tuple[int, ...]:
    """Cache topology-only N/CA/C lookup; coordinates never enter the key."""
    indices = [-1] * (num_residues * 3)
    for atom_index, residue_index in enumerate(atom_to_residue):
        atom_name = atom_names[atom_index]
        if atom_name in FRAME_ATOMS:
            slot = FRAME_ATOMS.index(atom_name)
            offset = residue_index * 3 + slot
            if indices[offset] < 0:
                indices[offset] = atom_index
    return tuple(indices)


@dataclass(frozen=True)
class ResidueFrameResult:
    """Residue-local frames and the atom indices used to construct them.

    ``frames[r]`` stores ``[e1, e2, e3]`` as columns.  For an invalid residue
    the frame is the identity and ``valid[r]`` is false; callers must use the
    mask before applying a geometric update.  Keeping a finite placeholder
    avoids NaNs leaking into batched diagnostics.
    """

    origins: Tensor
    frames: Tensor
    valid: Tensor
    atom_indices: Tensor

    def __post_init__(self) -> None:
        if self.origins.ndim != 2 or tuple(self.origins.shape[-1:]) != (3,):
            raise ValueError(f"origins must have shape [Nr, 3], got {tuple(self.origins.shape)}")
        if self.frames.ndim != 3 or tuple(self.frames.shape[-2:]) != (3, 3):
            raise ValueError(f"frames must have shape [Nr, 3, 3], got {tuple(self.frames.shape)}")
        if self.frames.shape[0] != self.origins.shape[0]:
            raise ValueError("origins and frames must have the same residue count")
        if self.valid.ndim != 1 or self.valid.shape[0] != self.origins.shape[0] or self.valid.dtype != torch.bool:
            raise ValueError("valid must be BoolTensor [Nr]")
        if self.atom_indices.shape != (self.origins.shape[0], 3) or self.atom_indices.dtype != torch.long:
            raise ValueError("atom_indices must be LongTensor [Nr, 3]")
        if not torch.isfinite(self.origins).all() or not torch.isfinite(self.frames).all():
            raise ValueError("frame outputs must be finite")

    @property
    def num_residues(self) -> int:
        return int(self.origins.shape[0])


def _check_inputs(protein_pos: Tensor, atom_to_residue: Tensor, atom_name: Sequence[str], num_residues: int) -> None:
    if not isinstance(protein_pos, torch.Tensor) or protein_pos.ndim != 2 or protein_pos.shape[-1] != 3:
        raise ValueError("protein_pos must be a tensor with shape [N, 3]")
    if not protein_pos.is_floating_point():
        raise TypeError("protein_pos must use a floating dtype")
    if not isinstance(atom_to_residue, torch.Tensor) or atom_to_residue.ndim != 1 or atom_to_residue.dtype != torch.long:
        raise TypeError("atom_to_residue must be a LongTensor [N]")
    if atom_to_residue.shape[0] != protein_pos.shape[0] or len(atom_name) != protein_pos.shape[0]:
        raise ValueError("protein_pos, atom_to_residue and atom_name must describe the same atom count")
    if num_residues <= 0:
        raise ValueError("num_residues must be positive")
    if atom_to_residue.numel():
        if int(atom_to_residue.min()) < 0 or int(atom_to_residue.max()) >= num_residues:
            raise ValueError("atom_to_residue contains an out-of-range residue id")


def build_residue_frames(
    protein_pos: Tensor,
    atom_to_residue: Tensor,
    atom_name: Sequence[str],
    *,
    num_residues: int,
    eps: float = 1.0e-6,
    atom_indices: Tensor | None = None,
) -> ResidueFrameResult:
    """Construct N–CA–C frames for all residues.

    The implementation follows the frozen definition exactly:

    ``e1 = normalize(C - CA)``, ``u = normalize(N - CA)``,
    ``e3 = normalize(e1 × u)``, ``e2 = e3 × e1``.

    Input coordinates are evaluated in float32 as required by the geometry
    contract.  Casting is differentiable, and no in-place operation is used on
    values gathered from ``protein_pos``.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    _check_inputs(protein_pos, atom_to_residue, atom_name, num_residues)
    coords = protein_pos.to(dtype=torch.float32)

    # The mapping is topology metadata, so constructing it on the host is
    # deterministic.  Gathering through a clamped index keeps the coordinate
    # path differentiable for valid atoms and supplies finite placeholders for
    # missing atoms.
    if atom_indices is None:
        cached = _cached_frame_atom_indices(
            tuple(atom_name),
            tuple(int(value) for value in atom_to_residue.detach().cpu().tolist()),
            num_residues,
        )
        indices = torch.tensor(cached, dtype=torch.long, device=coords.device).reshape(num_residues, 3)
    else:
        if atom_indices.dtype != torch.long or atom_indices.shape != (num_residues, 3):
            raise ValueError("atom_indices must be LongTensor [Nr, 3]")
        if atom_indices.device != coords.device:
            raise ValueError("atom_indices must share the coordinate device")
        indices = atom_indices

    safe_indices = indices.clamp_min(0)
    n_pos = coords[safe_indices[:, 0]]
    ca_pos = coords[safe_indices[:, 1]]
    c_pos = coords[safe_indices[:, 2]]
    has_atoms = indices.ge(0).all(dim=1)
    has_ca = indices[:, 1].ge(0)

    ca_finite = torch.isfinite(ca_pos).all(dim=1)
    n_finite = torch.isfinite(n_pos).all(dim=1)
    c_finite = torch.isfinite(c_pos).all(dim=1)
    vector_c = c_pos - ca_pos
    vector_n = n_pos - ca_pos
    norm_c = torch.linalg.vector_norm(vector_c, dim=1)
    norm_n = torch.linalg.vector_norm(vector_n, dim=1)
    e1 = vector_c / norm_c.clamp_min(eps).unsqueeze(1)
    unit_n = vector_n / norm_n.clamp_min(eps).unsqueeze(1)
    cross = torch.cross(e1, unit_n, dim=1)
    norm_cross = torch.linalg.vector_norm(cross, dim=1)
    e3 = cross / norm_cross.clamp_min(eps).unsqueeze(1)
    e2 = torch.cross(e3, e1, dim=1)
    frames = torch.stack((e1, e2, e3), dim=-1)

    valid = (
        has_atoms
        & ca_finite
        & n_finite
        & c_finite
        & torch.isfinite(norm_c)
        & torch.isfinite(norm_n)
        & torch.isfinite(norm_cross)
        & (norm_c > eps)
        & (norm_n > eps)
        & (norm_cross > eps)
    )
    identity = torch.eye(3, dtype=coords.dtype, device=coords.device).expand(num_residues, -1, -1)
    frames = torch.where(valid[:, None, None], frames, identity)
    zero = torch.zeros_like(ca_pos)
    origins = torch.where(has_ca[:, None] & ca_finite[:, None], ca_pos, zero)
    return ResidueFrameResult(origins=origins, frames=frames, valid=valid, atom_indices=indices)


def frame_orthogonality_error(frames: Tensor) -> Tensor:
    """Return the maximum absolute ``F.T @ F - I`` entry."""

    if frames.ndim != 3 or frames.shape[-2:] != (3, 3):
        raise ValueError("frames must have shape [Nr, 3, 3]")
    identity = torch.eye(3, dtype=frames.dtype, device=frames.device)
    return (frames.transpose(-1, -2) @ frames - identity).abs().max()


__all__ = ["FRAME_ATOMS", "ResidueFrameResult", "build_residue_frames", "frame_orthogonality_error"]
