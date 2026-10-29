"""Oracle coordinate reconstruction for the independent PocketDiff model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .bridge import (
    OracleReconstructionMetrics,
    apply_fractional_update,
    oracle_reconstruction_metrics,
    remaining_transform_current_to_holo,
)
from .chi import apply_chi_updates, periodic_chi_delta
from .current_state import CurrentChiState, build_current_chi_state
from .frames import build_residue_frames


@dataclass(frozen=True)
class OracleReconstruction:
    """Rigid-only and rigid+χ oracle outputs for one complex."""

    rigid_only_positions: torch.Tensor
    rigid_chi_positions: torch.Tensor
    rigid_metrics: OracleReconstructionMetrics
    rigid_chi_metrics: OracleReconstructionMetrics
    current_chi: CurrentChiState
    applied_chi: torch.Tensor
    supervision_mask: torch.Tensor
    training_safe_mask: torch.Tensor
    frame_valid: torch.Tensor


def oracle_rigid_chi_reconstruction(
    apo_pos: torch.Tensor,
    holo_pos: torch.Tensor,
    atom_to_residue: torch.Tensor,
    atom_names: Sequence[str],
    residue_names: Sequence[str],
    frame_valid: torch.Tensor,
) -> OracleReconstruction:
    """Apply an exact residue rigid transform, then an oracle χ delta.

    The current χ state and geometry mask are computed from ``apo_pos`` only.
    Holo angles are recomputed from coordinates, never read from historical
    caches. Named-atom supervision is used for this reconstruction diagnostic.
    ``training_safe_mask`` separately excludes ambiguous slots until a full
    equivalent-target loss is implemented; it does not disable inference
    rotations. This reference-based residual is not a proven optimal floor.
    """

    if apo_pos.shape != holo_pos.shape or apo_pos.ndim != 2 or apo_pos.shape[-1] != 3:
        raise ValueError("apo_pos and holo_pos must both have shape [N, 3]")
    if atom_to_residue.dtype != torch.long or atom_to_residue.shape != (apo_pos.shape[0],):
        raise ValueError("atom_to_residue must be LongTensor [N]")
    if frame_valid.dtype != torch.bool or frame_valid.shape != (len(residue_names),):
        raise ValueError("frame_valid must be BoolTensor [Nr]")
    if any(t.device != apo_pos.device for t in (holo_pos, atom_to_residue, frame_valid)):
        raise ValueError("oracle tensors must be on the same device")
    current_chi = build_current_chi_state(apo_pos, atom_names, atom_to_residue, residue_names)
    target_chi = build_current_chi_state(holo_pos, atom_names, atom_to_residue, residue_names)

    apo_frames = build_residue_frames(
        apo_pos, atom_to_residue, atom_names, num_residues=len(residue_names)
    )
    holo_frames = build_residue_frames(
        holo_pos, atom_to_residue, atom_names, num_residues=len(residue_names)
    )
    valid = frame_valid & apo_frames.valid & holo_frames.valid
    remaining = remaining_transform_current_to_holo(
        apo_frames.origins,
        apo_frames.frames,
        holo_frames.origins,
        holo_frames.frames,
        frame_valid=valid,
    )
    rigid_only = apply_fractional_update(
        apo_pos,
        atom_to_residue,
        apo_frames.origins,
        apo_frames.frames,
        remaining.translation_local,
        remaining.rotvec_local,
        remaining_steps=1,
        frame_valid=valid,
    )

    supervision_mask = (
        target_chi.geometry_rotatable_mask
        & current_chi.geometry_rotatable_mask
        & valid[:, None]
    )
    chi_delta = periodic_chi_delta(current_chi.angles, target_chi.angles, supervision_mask)
    chi_update = apply_chi_updates(
        rigid_only,
        current_chi.axis_start,
        current_chi.axis_end,
        current_chi.downstream_atom_mask,
        chi_delta,
        valid=supervision_mask,
    )
    rigid_metrics = oracle_reconstruction_metrics(
        rigid_only, holo_pos, atom_to_residue, atom_names, valid
    )
    rigid_chi_metrics = oracle_reconstruction_metrics(
        chi_update.positions, holo_pos, atom_to_residue, atom_names, valid
    )
    return OracleReconstruction(
        rigid_only_positions=rigid_only,
        rigid_chi_positions=chi_update.positions,
        rigid_metrics=rigid_metrics,
        rigid_chi_metrics=rigid_chi_metrics,
        current_chi=current_chi,
        applied_chi=chi_update.applied_chi,
        supervision_mask=supervision_mask,
        training_safe_mask=supervision_mask & ~current_chi.ambiguous_chi_mask,
        frame_valid=valid,
    )


__all__ = ["OracleReconstruction", "oracle_rigid_chi_reconstruction"]
