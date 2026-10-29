"""Differentiable residue SE(3)+chi step with no ligand sampler dependency."""
from typing import Optional, Sequence, Union

import torch

from pocketdiff.data.schema import PocketDiffPrediction, PocketStepOutput
from .chi import apply_chi_updates
from .chi import ChiUpdateMetadata
from .current_state import build_current_chi_state
from .frames import ResidueFrameResult, build_residue_frames
from .so3 import so3_exp


def apply_joint_update(
    protein_pos: torch.Tensor,
    apo_pos_ref: torch.Tensor,
    atom_to_residue: torch.Tensor,
    atom_names: Sequence[str],
    residue_names: Sequence[str],
    prediction: PocketDiffPrediction,
    *,
    remaining_steps: Union[int, torch.Tensor],
    frame_atom_indices: Optional[torch.Tensor] = None,
    chi_metadata: Optional[ChiUpdateMetadata] = None,
    apo_frame: Optional[ResidueFrameResult] = None,
) -> PocketStepOutput:
    """Apply 1/n remaining translation, rotation and principal chi delta.

    Angles/topology/frames are computed from current coordinates. Apo supplies
    reference-frame validity only. No holo or supervision mask is accepted.
    SE(3) uses an increment formula to be bitwise stationary and differentiable
    at zero motion; a zero-output bypass would kill coordinate-loss gradients.
    """
    if protein_pos.dtype != torch.float32 or apo_pos_ref.dtype != torch.float32:
        raise TypeError("joint solver currently requires float32 protein coordinates")
    if protein_pos.shape != apo_pos_ref.shape or not torch.isfinite(apo_pos_ref).all():
        raise ValueError("apo_pos_ref must match protein_pos and be finite")
    if prediction.remaining_chi is None:
        raise ValueError("joint update requires remaining_chi")
    nr = len(residue_names)
    if prediction.frame_valid.shape != (nr,):
        raise ValueError("prediction and topology residue counts differ")
    tensors = (apo_pos_ref, atom_to_residue, prediction.remaining_translation_local,
               prediction.remaining_rotvec_local, prediction.remaining_chi, prediction.frame_valid)
    if any(v.device != protein_pos.device for v in tensors):
        raise ValueError("joint update tensors must share a device")
    if any(v.dtype != torch.float32 for v in tensors[2:5]):
        raise TypeError("joint predictions must be float32")
    if isinstance(remaining_steps, bool):
        raise TypeError("remaining_steps must be an integer or LongTensor")
    if isinstance(remaining_steps, int):
        steps = torch.full((nr,), remaining_steps, dtype=torch.long, device=protein_pos.device)
    elif isinstance(remaining_steps, torch.Tensor):
        if remaining_steps.dtype != torch.long or remaining_steps.shape != (nr,):
            raise ValueError("remaining_steps must be LongTensor [Nr]")
        steps = remaining_steps.to(protein_pos.device)
    else:
        raise TypeError("remaining_steps must be an integer or LongTensor")
    if bool(((steps < 1) | (steps > 20)).any()):
        raise ValueError("remaining_steps must lie in [1, 20]")
    current = build_current_chi_state(
        protein_pos, atom_names, atom_to_residue, residue_names, metadata=chi_metadata
    )
    frames = build_residue_frames(
        protein_pos, atom_to_residue, atom_names, num_residues=nr,
        atom_indices=frame_atom_indices,
    )
    if apo_frame is None:
        apo_frames = build_residue_frames(
            apo_pos_ref, atom_to_residue, atom_names, num_residues=nr,
            atom_indices=frame_atom_indices,
        )
    else:
        if not isinstance(apo_frame, ResidueFrameResult):
            raise TypeError("apo_frame must be a ResidueFrameResult")
        if (
            apo_frame.num_residues != nr
            or apo_frame.origins.device != protein_pos.device
            or apo_frame.frames.device != protein_pos.device
            or apo_frame.valid.device != protein_pos.device
        ):
            raise ValueError("apo_frame is incompatible with the joint update")
        apo_frames = apo_frame
    valid = frames.valid & apo_frames.valid & prediction.frame_valid
    denominator = steps.to(protein_pos.dtype)[:, None]
    translation = torch.where(valid[:, None], prediction.remaining_translation_local / denominator, 0.)
    rotation = torch.where(valid[:, None], prediction.remaining_rotvec_local / denominator, 0.)
    active = current.geometry_rotatable_mask & valid[:, None]
    # Wrap BEFORE dividing; e.g. a +2pi representation is a zero remaining
    # angle, not a spurious pi half-step.
    principal = torch.atan2(prediction.remaining_chi.sin(), prediction.remaining_chi.cos())
    chi_delta = torch.where(active, principal / denominator, 0.)

    rotation_increment_local = so3_exp(rotation) - torch.eye(3, dtype=protein_pos.dtype, device=protein_pos.device)
    rotation_increment = frames.frames @ rotation_increment_local @ frames.frames.transpose(-1, -2)
    translation_global = (frames.frames @ translation.unsqueeze(-1)).squeeze(-1)
    relative = protein_pos - frames.origins[atom_to_residue]
    rigid = protein_pos + torch.bmm(
        rotation_increment[atom_to_residue], relative.unsqueeze(-1),
    ).squeeze(-1) + translation_global[atom_to_residue]
    chi_update = apply_chi_updates(rigid, current.axis_start, current.axis_end,
                                   current.downstream_atom_mask, chi_delta, valid=active)
    updated = chi_update.positions
    next_frames = build_residue_frames(
        updated, atom_to_residue, atom_names, num_residues=nr,
        atom_indices=frame_atom_indices,
    )
    next_chi = build_current_chi_state(
        updated, atom_names, atom_to_residue, residue_names, metadata=chi_metadata
    )
    if bool((active & ~chi_update.valid).any()):
        raise ValueError("requested chi update lost a valid axis")
    return PocketStepOutput(
        protein_pos_next=updated, prediction=prediction,
        applied_translation_local=translation, applied_rotvec_local=rotation,
        applied_chi=chi_update.applied_chi,
        diagnostics=dict(remaining_steps=steps, frame_valid=valid,
                         frame_valid_next=next_frames.valid, current_chi=current.angles,
                         chi_next=next_chi.angles, chi_rotatable_mask=active,
                         chi_rotatable_mask_next=next_chi.geometry_rotatable_mask),
    )
