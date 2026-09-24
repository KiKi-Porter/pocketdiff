"""Inference-side residue state derived from the current coordinates.

This module deliberately accepts only the current protein coordinates and
topology.  It never needs a holo structure or a holo-derived supervision
mask, which makes its output safe to expose during autonomous inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .chi import ChiUpdateMetadata, build_chi_update_metadata, extract_chi_angles


CURRENT_CHI_VERSION = "current-chi-v1-canonical-heavy-atoms"

# These slots require equivalent-target supervision before training. The four
# planar pi-periodic cases follow AlphaFold residue_constants.chi_pi_periodic;
# LEU/VAL branched terminal aliases are conservatively excluded too, not
# treated as pi-periodic. This mask never disables geometry updates.
AMBIGUOUS_CHI_SLOTS = {"ASP": 1, "GLU": 2, "PHE": 1, "TYR": 1, "LEU": 1, "VAL": 0}


@dataclass(frozen=True)
class CurrentChiState:
    """Current side-chain angles and inference-safe rotatable mask."""

    angles: torch.Tensor
    geometry_rotatable_mask: torch.Tensor
    axis_start: torch.Tensor
    axis_end: torch.Tensor
    downstream_atom_mask: torch.Tensor
    ambiguous_chi_mask: torch.Tensor

    def __post_init__(self) -> None:
        if not self.angles.is_floating_point() or self.angles.ndim != 2 or self.angles.shape[-1] != 5:
            raise ValueError("angles must have shape [Nr, 5]")
        if self.geometry_rotatable_mask.dtype != torch.bool or self.geometry_rotatable_mask.shape != self.angles.shape:
            raise ValueError("geometry_rotatable_mask must be BoolTensor [Nr, 5]")
        if self.axis_start.dtype != torch.long or self.axis_end.dtype != torch.long:
            raise TypeError("axis indices must be LongTensor")
        if self.axis_start.shape != self.angles.shape or self.axis_end.shape != self.angles.shape:
            raise ValueError("axis tensors must have shape [Nr, 5]")
        ChiUpdateMetadata(self.axis_start, self.axis_end, self.downstream_atom_mask,
                          self.geometry_rotatable_mask)
        if self.ambiguous_chi_mask.dtype != torch.bool or self.ambiguous_chi_mask.shape != self.angles.shape:
            raise ValueError("ambiguous_chi_mask must be BoolTensor [Nr, 5]")
        if any(t.device != self.angles.device for t in (
            self.axis_start, self.axis_end, self.downstream_atom_mask,
            self.geometry_rotatable_mask, self.ambiguous_chi_mask,
        )):
            raise ValueError("current chi tensors must be on the same device")
        if not torch.isfinite(self.angles).all():
            raise ValueError("angles must be finite")


def build_current_chi_state(
    positions: torch.Tensor,
    atom_names: Sequence[str],
    atom_to_residue: torch.Tensor,
    residue_names: Sequence[str],
    *,
    metadata: ChiUpdateMetadata | None = None,
) -> CurrentChiState:
    """Extract χ angles and topology using only the current structure.

    ``geometry_rotatable_mask`` is based on the current atom inventory and
    standard residue topology.  It intentionally does not compare against a
    holo structure and does not use a supervision mask.
    """

    if positions.ndim != 2 or positions.shape[-1] != 3 or not positions.is_floating_point():
        raise ValueError("positions must be floating [N, 3]")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must be finite")
    if positions.device != atom_to_residue.device:
        raise ValueError("positions and atom_to_residue must be on the same device")
    metadata = metadata or build_chi_update_metadata(atom_names, atom_to_residue, residue_names)
    angles, angle_valid = extract_chi_angles(
        positions, atom_names, atom_to_residue, residue_names, metadata=metadata
    )
    geometry_mask = metadata.valid & angle_valid
    ambiguous = torch.zeros_like(geometry_mask)
    for index, name in enumerate(residue_names):
        slot = AMBIGUOUS_CHI_SLOTS.get(str(name).upper())
        if slot is not None:
            ambiguous[index, slot] = True
    angles = torch.where(geometry_mask, angles, torch.zeros_like(angles))
    return CurrentChiState(
        angles=angles,
        geometry_rotatable_mask=geometry_mask,
        axis_start=metadata.axis_start,
        axis_end=metadata.axis_end,
        downstream_atom_mask=metadata.downstream_atom_mask,
        ambiguous_chi_mask=ambiguous,
    )


__all__ = ["CURRENT_CHI_VERSION", "CurrentChiState", "build_current_chi_state"]
