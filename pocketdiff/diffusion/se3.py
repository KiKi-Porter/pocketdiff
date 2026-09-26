"""Shared local SE(3) conventions for diffusion targets and sampling.

Coordinates are row vectors.  A residue frame stores local basis vectors as
columns, so a local translation ``u`` becomes ``u @ frame.T`` in global
coordinates and a local rotation ``R`` becomes ``frame @ R @ frame.T``.
Atoms are updated around the current residue origin with ``x @ R_global.T``.
"""
from __future__ import annotations

from typing import Optional, Union

import torch

from pocketdiff.geometry.so3 import so3_exp

Tensor = torch.Tensor


def apply_local_se3_update(
    protein_pos: Tensor,
    atom_to_residue: Tensor,
    current_origins: Tensor,
    current_frames: Tensor,
    translation_local: Tensor,
    rotvec_local: Tensor,
    *,
    fraction: Union[float, Tensor] = 1.0,
    frame_valid: Optional[Tensor] = None,
) -> Tensor:
    """Apply one local residue transform to row-vector atom coordinates."""
    if protein_pos.ndim != 2 or protein_pos.shape[-1] != 3:
        raise ValueError("protein_pos must have shape [N, 3]")
    if atom_to_residue.dtype != torch.long or atom_to_residue.shape != (protein_pos.shape[0],):
        raise ValueError("atom_to_residue must be LongTensor [N]")
    if translation_local.shape != rotvec_local.shape or translation_local.ndim != 2:
        raise ValueError("local transforms must have shape [Nr, 3]")
    nr = translation_local.shape[0]
    if current_origins.shape != (nr, 3) or current_frames.shape != (nr, 3, 3):
        raise ValueError("current residue geometry has incompatible shape")
    if frame_valid is None:
        frame_valid = torch.ones(nr, dtype=torch.bool, device=protein_pos.device)
    if frame_valid.dtype != torch.bool or frame_valid.shape != (nr,):
        raise ValueError("frame_valid must be BoolTensor [Nr]")
    frac = torch.as_tensor(fraction, dtype=torch.float32, device=protein_pos.device)
    if frac.numel() not in (1, nr):
        raise ValueError("fraction must be scalar or one value per residue")
    frac = frac.reshape(-1)
    if frac.numel() == 1:
        frac = frac.expand(nr)
    if not torch.isfinite(frac).all():
        raise ValueError("fraction must be finite")

    frames = current_frames.to(dtype=torch.float32)
    origins = current_origins.to(dtype=torch.float32)
    positions = protein_pos.to(dtype=torch.float32)
    local_t = translation_local.to(dtype=torch.float32) * frac[:, None]
    local_r = rotvec_local.to(dtype=torch.float32) * frac[:, None]
    global_r = frames @ so3_exp(local_r) @ frames.transpose(-1, -2)
    global_t = torch.bmm(local_t.unsqueeze(1), frames.transpose(-1, -2)).squeeze(1)
    rid = atom_to_residue
    atom_origin = origins[rid]
    centered = (positions - atom_origin).unsqueeze(1)
    updated = torch.bmm(centered, global_r[rid].transpose(-1, -2)).squeeze(1)
    updated = updated + atom_origin + global_t[rid]
    zero_update = (local_t == 0.0).all(dim=-1) & (local_r == 0.0).all(dim=-1)
    updated = torch.where(zero_update[rid].unsqueeze(1), positions, updated)
    return torch.where(frame_valid[rid].unsqueeze(1), updated, positions)


__all__ = ["apply_local_se3_update"]
