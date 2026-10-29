"""Online autonomous-state refresh for self-state training."""

from dataclasses import dataclass

import torch

from .clean import CleanBatch
from .self_state import SelfStateBatch, build_self_state_batch


@dataclass(frozen=True)
class OnlineTrajectory:
    """Detached model-owned states that can be used as self-state inputs."""

    positions: torch.Tensor
    frame_valid: torch.Tensor
    update_valid: torch.Tensor

    def __post_init__(self):
        if self.positions.ndim != 3 or self.positions.shape[0] != 21 or self.positions.shape[-1] != 3:
            raise ValueError('positions must have shape [21, Np, 3]')
        if self.frame_valid.ndim != 2 or self.frame_valid.shape[0] != 21:
            raise ValueError('frame_valid must have shape [21, Nr]')
        if self.update_valid.ndim != 2 or self.update_valid.shape[0] != 20:
            raise ValueError('update_valid must have shape [20, Nr]')
        if self.frame_valid.shape[1] != self.update_valid.shape[1]:
            raise ValueError('frame and update masks have different residue counts')
        if self.positions.device.type != 'cpu' or self.frame_valid.device.type != 'cpu' or self.update_valid.device.type != 'cpu':
            raise ValueError('online trajectories currently use CPU tensors')
        if not torch.isfinite(self.positions).all():
            raise ValueError('online trajectory positions must be finite')
        if self.frame_valid.dtype != torch.bool or self.update_valid.dtype != torch.bool:
            raise TypeError('online trajectory masks must be BoolTensor')


@torch.no_grad()
def refresh_online_trajectory(model, clean: CleanBatch) -> OnlineTrajectory:
    """Regenerate detached autonomous states from the current model.

    The rollout receives only apo/reference/features/clean ligand fields. It
    does not receive holo coordinates or labels. Existing mode, RNG and
    endpoint tensors are protected by the shared clean-rollout contract.
    """
    if type(clean) is not CleanBatch:
        raise TypeError('clean must be endpoint CleanBatch')
    from pocketdiff.sampling.clean_rollout import INFERENCE_FIELDS, run_clean_rollout
    inputs = {name: getattr(clean, name) for name in INFERENCE_FIELDS}
    trace = run_clean_rollout(model, inputs)
    if not torch.equal(trace.positions[0], clean.apo_pos_ref):
        raise RuntimeError('online rollout step 0 is not the apo endpoint')
    return OnlineTrajectory(trace.positions.detach().cpu().clone(),
                            trace.frame_valid.detach().cpu().clone(),
                            trace.update_valid.detach().cpu().clone())


def build_latest_self_state(clean: CleanBatch, trajectory: OnlineTrajectory,
                            pocket_k: torch.Tensor) -> SelfStateBatch:
    """Build physical remaining labels from the most recently refreshed states."""
    if not isinstance(trajectory, OnlineTrajectory):
        raise TypeError('trajectory must be an OnlineTrajectory')
    if trajectory.positions.shape[1] != clean.protein_pos.shape[0]:
        raise ValueError('trajectory atom count differs from clean batch')
    return build_self_state_batch(clean, trajectory.positions, pocket_k)


__all__ = ['OnlineTrajectory', 'build_latest_self_state', 'refresh_online_trajectory']
