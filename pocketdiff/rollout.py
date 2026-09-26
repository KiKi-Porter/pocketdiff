"""Independent apo-start, clean-ligand joint rollout. No target is accepted."""
from dataclasses import dataclass, replace

import torch

from pocketdiff.inference import PocketInputs, predict_joint_step
from pocketdiff.models import PocketDiffModel


@dataclass(frozen=True)
class JointRollout:
    positions: torch.Tensor  # [21, Np, 3], including the initial apo state
    frame_valid: torch.Tensor  # fixed apo-derived mask, not a holo mask


@torch.no_grad()
def rollout_joint_from_apo(model: PocketDiffModel, inputs: PocketInputs) -> JointRollout:
    """Keep ligand fixed and run k=0..19 using only the previous prediction.

    No oracle teacher forcing, target-dependent stopping, or realignment.
    Check frame preservation without silently shrinking the evaluation mask.
    The caller's model mode is restored, including on failure.
    """
    values = inputs.model_kwargs()
    if (not torch.equal(inputs.protein_pos, inputs.apo_pos_ref)
            or bool((inputs.pocket_k != 0).any())
            or bool((inputs.targetdiff_t != 199).any())):
        raise ValueError('rollout requires apo at k=0/t=199')
    valid = values['frame_valid']
    was_training = model.training
    model.eval()
    positions = [inputs.protein_pos.detach().clone()]
    try:
        for k in range(20):
            state = replace(inputs, protein_pos=positions[-1],
                            pocket_k=torch.full_like(inputs.pocket_k, k),
                            targetdiff_t=torch.full_like(inputs.targetdiff_t, 199 - 10 * k))
            output = predict_joint_step(model, state)
            if not torch.isfinite(output.protein_pos_next).all():
                raise FloatingPointError('nonfinite autonomous joint state at k=%d' % k)
            if bool((valid & ~output.diagnostics['frame_valid_next']).any()):
                raise ValueError('autonomous joint step lost a valid frame at k=%d' % k)
            positions.append(output.protein_pos_next.detach().clone())
    finally:
        model.train(was_training)
    return JointRollout(torch.stack(positions), valid.clone())


__all__ = ['JointRollout', 'rollout_joint_from_apo']
