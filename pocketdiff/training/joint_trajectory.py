"""Teacher-forced rigid+chi coordinates; experimental holo stays label-side."""
from dataclasses import dataclass, replace

import torch

from pocketdiff.data.apo2mol_adapter import AA_NAMES
from pocketdiff.geometry.bridge import build_bridge_state
from pocketdiff.geometry.chi import apply_chi_updates, periodic_chi_delta
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.inference import PocketInputs
from .coordinate_step import NextXYZBatch


@dataclass(frozen=True)
class JointReferenceTrajectory:
    """21 reference states, with inference and supervision masks separated.

    The endpoint preserves apo internal geometry; it is an oracle rigid+chi
    reconstruction, not necessarily experimental holo. Do not feed this
    trajectory to autonomous inference.
    """
    apo_inputs: PocketInputs
    positions: torch.Tensor
    reference_frame_valid: torch.Tensor
    supervision_frame_valid: torch.Tensor
    ambiguous_residues: torch.Tensor

    def batch_at(self, pocket_k: torch.Tensor) -> NextXYZBatch:
        expected = self.apo_inputs.pocket_k
        if (pocket_k.dtype != torch.long or pocket_k.shape != expected.shape
                or pocket_k.device != expected.device):
            raise ValueError('pocket_k must be LongTensor [B] on the input device')
        if bool(((pocket_k < 0) | (pocket_k > 19)).any()):
            raise ValueError('pocket_k must lie in [0,19]')
        k = pocket_k.clone()
        atom_k = k[self.apo_inputs.batch_protein]
        atoms = torch.arange(atom_k.numel(), device=atom_k.device)
        inputs = replace(self.apo_inputs, protein_pos=self.positions[atom_k, atoms],
                         pocket_k=k, targetdiff_t=199 - 10 * k)
        return NextXYZBatch(inputs, self.positions[atom_k + 1, atoms],
                            self.supervision_frame_valid)


@torch.no_grad()
def build_joint_reference(apo_inputs: PocketInputs,
                          protein_pos_holo: torch.Tensor) -> JointReferenceTrajectory:
    """Interpolate apo rigid frames and principal chi differences at k/20.

    Holo/masks never enter PocketInputs. Conservatively exclude ambiguous
    residues and missing target torsions from coordinate supervision only.
    """
    apo_inputs.model_kwargs()
    apo = apo_inputs.apo_pos_ref
    if (not torch.equal(apo_inputs.protein_pos, apo)
            or bool((apo_inputs.pocket_k != 0).any())
            or bool((apo_inputs.targetdiff_t != 199).any())):
        raise ValueError('reference construction requires apo at k=0/t=199')
    holo = protein_pos_holo
    if (holo.shape != apo.shape or holo.dtype != torch.float32
            or holo.device != apo.device or not torch.isfinite(holo).all()):
        raise ValueError('holo must be finite float32 [Np,3] on the input device')
    ids, names = apo_inputs.atom_to_residue_global, apo_inputs.protein_atom_name
    residues = [AA_NAMES[i] for i in apo_inputs.residue_type.cpu().tolist()]
    frames = build_residue_frames(apo, ids, names, num_residues=len(residues))
    target_frames = build_residue_frames(holo, ids, names, num_residues=len(residues))
    valid = frames.valid & target_frames.valid
    current = build_current_chi_state(apo, names, ids, residues)
    target = build_current_chi_state(holo, names, ids, residues)
    chi_valid = current.geometry_rotatable_mask & target.geometry_rotatable_mask & valid[:, None]
    delta = periodic_chi_delta(current.angles, target.angles, chi_valid)
    ambiguous = current.ambiguous_chi_mask.any(-1)
    missing_target = (current.geometry_rotatable_mask & ~target.geometry_rotatable_mask).any(-1)
    supervision = valid & ~ambiguous & ~missing_target
    if not supervision.any():
        raise ValueError('reference has no safe supervised residue')
    positions = [apo.clone()]
    for k in range(1, 21):
        rigid = build_bridge_state(
            apo, ids, frames.origins, frames.frames, target_frames.origins,
            target_frames.frames, fraction=k / 20., frame_valid=valid,
        ).protein_pos
        joint = apply_chi_updates(rigid, current.axis_start, current.axis_end,
                                  current.downstream_atom_mask, delta * (k / 20.), valid=chi_valid)
        positions.append(joint.positions)
        next_frames = build_residue_frames(joint.positions, ids, names, num_residues=len(residues))
        if bool((valid & ~next_frames.valid).any()):
            raise ValueError('reference interpolation invalidated a residue frame')
    return JointReferenceTrajectory(apo_inputs, torch.stack(positions), valid, supervision, ambiguous)


__all__ = ['JointReferenceTrajectory', 'build_joint_reference']
