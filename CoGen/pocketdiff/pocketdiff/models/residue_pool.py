"""Global residue pooling helpers."""

from __future__ import annotations

import torch


def scatter_mean_residue(values: torch.Tensor, atom_to_residue_global: torch.Tensor, num_residues: int) -> torch.Tensor:
    if values.ndim != 2 or atom_to_residue_global.ndim != 1 or values.shape[0] != atom_to_residue_global.shape[0]:
        raise ValueError("values and atom_to_residue_global have incompatible shapes")
    if atom_to_residue_global.dtype != torch.long:
        raise TypeError("atom_to_residue_global must be LongTensor")
    if num_residues <= 0 or atom_to_residue_global.numel() == 0:
        raise ValueError("residue pooling requires positive residues and at least one atom")
    if int(atom_to_residue_global.min()) < 0 or int(atom_to_residue_global.max()) >= num_residues:
        raise ValueError("atom_to_residue_global contains an out-of-range id")
    pooled = torch.zeros((num_residues, values.shape[1]), dtype=values.dtype, device=values.device)
    pooled.index_add_(0, atom_to_residue_global, values)
    counts = torch.zeros(num_residues, dtype=values.dtype, device=values.device)
    counts.index_add_(0, atom_to_residue_global, torch.ones(atom_to_residue_global.shape[0], dtype=values.dtype, device=values.device))
    if bool((counts == 0).any()):
        raise ValueError("atom_to_residue_global does not cover every residue")
    return pooled / counts[:, None]


__all__ = ["scatter_mean_residue"]
