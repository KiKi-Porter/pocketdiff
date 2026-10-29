"""Autonomous 20-step pocket inference with a fixed clean ligand condition.

This module accepts no holo coordinates, target labels or teacher-forced states.
It shares the existing fractional geometry solver but does not run TargetDiff.
"""

from dataclasses import dataclass

import torch

from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models import PocketDiffModel


INFERENCE_FIELDS = (
    'apo_pos_ref', 'protein_feature', 'atom_to_residue_global', 'residue_type',
    'batch_protein', 'batch_residue', 'ligand_pos', 'ligand_v', 'batch_ligand',
    'protein_atom_name',
)


@dataclass(frozen=True)
class CleanRollout:
    positions: torch.Tensor  # [21, Np, 3], initial apo followed by 20 predictions
    frame_valid: torch.Tensor  # [21, Nr], geometry only
    update_valid: torch.Tensor  # [20, Nr], geometry & model & apo mask
    applied_translation_local: torch.Tensor  # [20, Nr, 3]
    applied_rotvec_local: torch.Tensor  # [20, Nr, 3]


@torch.no_grad()
def run_clean_rollout(model, inputs):
    """Run in eval/no-grad, restore mode, and reject in-place input mutation.

    ``inputs`` must contain exactly INFERENCE_FIELDS. Frames/masks and the
    k=0..19 schedule are derived here. Each next state comes only from the
    previous predicted state; no recentering, alignment or state reset occurs.
    """
    if set(inputs) != set(INFERENCE_FIELDS):
        raise ValueError('rollout inputs must contain exactly INFERENCE_FIELDS (no targets or current state)')
    fixed = {name: value.detach().clone() if isinstance(value, torch.Tensor) else list(value)
             for name, value in inputs.items()}
    if any(not torch.isfinite(v).all() for v in fixed.values()
           if isinstance(v, torch.Tensor) and v.is_floating_point()):
        raise ValueError('rollout inputs must be finite')
    geometry = dict(atom_to_residue=fixed['atom_to_residue_global'],
                    atom_name=fixed['protein_atom_name'], num_residues=fixed['residue_type'].numel())
    current = fixed['apo_pos_ref'].clone()
    frames = build_residue_frames(current, **geometry)
    apo_valid = frames.valid.clone()
    graph_count = int(fixed['batch_protein'].max()) + 1
    k_graph = torch.zeros(graph_count, dtype=torch.long, device=current.device)
    PocketDiffModel._validate_batch(protein_pos=current, frame_valid=apo_valid,
                                   pocket_k=k_graph, targetdiff_t=199-10*k_graph, **fixed)
    positions, valid_frames = [current.clone()], [frames.valid.clone()]
    valid_updates, translations, rotations = [], [], []
    was_training = model.training
    model.eval()
    try:
        for k in range(20):
            k_graph = torch.full_like(k_graph, k)
            current_before = current.clone()
            mask = apo_valid.clone()
            pred = model(protein_pos=current, frame_valid=mask, pocket_k=k_graph,
                         targetdiff_t=199-10*k_graph, **fixed)
            if not torch.equal(current, current_before) or not torch.equal(mask, apo_valid):
                raise RuntimeError('model mutated current coordinates or reference frame mask')
            for name in INFERENCE_FIELDS:
                same = (torch.equal(fixed[name], inputs[name]) if isinstance(fixed[name], torch.Tensor)
                        else fixed[name] == list(inputs[name]))
                if not same:
                    raise RuntimeError('model mutated fixed input: ' + name)
            if not bool((k_graph == k).all()):
                raise RuntimeError('model mutated pocket schedule')
            if pred.remaining_chi is not None:
                raise ValueError('clean MVP rollout does not support chi updates')
            if pred.frame_valid.shape != apo_valid.shape:
                raise ValueError('prediction residue count differs from rollout inputs')
            if not (torch.isfinite(pred.remaining_translation_local).all()
                    and torch.isfinite(pred.remaining_rotvec_local).all()):
                raise FloatingPointError('non-finite prediction at k=%d' % k)
            valid = apo_valid & frames.valid & pred.frame_valid
            remaining = 20-k
            current = apply_fractional_update(
                current, fixed['atom_to_residue_global'], frames.origins, frames.frames,
                pred.remaining_translation_local, pred.remaining_rotvec_local,
                remaining_steps=remaining, frame_valid=valid)
            if not torch.isfinite(current).all():
                raise FloatingPointError('non-finite coordinates at k=%d' % k)
            frames = build_residue_frames(current, **geometry)
            positions.append(current.clone())
            valid_frames.append(frames.valid.clone())
            valid_updates.append(valid.clone())
            translations.append(torch.where(valid[:, None], pred.remaining_translation_local/remaining,
                                            torch.zeros_like(pred.remaining_translation_local)))
            rotations.append(torch.where(valid[:, None], pred.remaining_rotvec_local/remaining,
                                         torch.zeros_like(pred.remaining_rotvec_local)))
    finally:
        model.train(was_training)
    return CleanRollout(torch.stack(positions), torch.stack(valid_frames), torch.stack(valid_updates),
                        torch.stack(translations), torch.stack(rotations))


__all__ = ['CleanRollout', 'INFERENCE_FIELDS', 'run_clean_rollout']
