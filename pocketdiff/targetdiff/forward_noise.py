"""Reference-anchored forward conditions using the loaded TargetDiff schedule."""

from typing import Optional

import torch

from .adapter import (
    TargetDiffAdapter, TargetDiffStepRandomness, _index_to_log_onehot,
    _randn_like, _sample_log_categorical,
)
from .state import TargetDiffState


@torch.no_grad()
def forward_noise_reference(
    adapter: TargetDiffAdapter,
    reference: TargetDiffState,
    t_graph: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    randomness: Optional[TargetDiffStepRandomness] = None,
) -> TargetDiffState:
    """Sample q(L_t | L_ref), retaining protein, atom order and apo offset.

    ``reference.ligand_pos`` must already be in the fixed apo-centered frame.
    Each call starts from the clean reference, never a previously noised state.
    ``t_graph`` has one independently selected time per graph. Even t=0 follows
    the checkpoint's first noise level; clean input is a separate condition.
    Random order is Gaussian position followed by categorical uniform.
    """
    adapter._validate_state(reference)
    if generator is not None and randomness is not None:
        raise ValueError("provide either generator or randomness, not both")
    if (t_graph.dtype != torch.long or t_graph.shape != (reference.num_graphs,)
            or t_graph.device != adapter.device):
        raise ValueError("t_graph must be LongTensor [num_graphs] on adapter device")
    if bool(((t_graph < 0) | (t_graph >= adapter.num_timesteps)).any()):
        raise ValueError("t_graph outside TargetDiff schedule")
    if randomness is not None:
        if randomness.position_noise.shape != reference.ligand_pos.shape:
            raise ValueError("randomness atom count must match reference ligand")
        if not randomness.position_noise.is_floating_point() or not randomness.categorical_uniform.is_floating_point():
            raise ValueError("randomness tensors must be floating point")
        noise = randomness.position_noise.to(reference.ligand_pos)
        uniform = randomness.categorical_uniform.to(device=adapter.device, dtype=torch.float32)
    else:
        noise = _randn_like(reference.ligand_pos, generator)
        uniform = None
    a = adapter.model.alphas_cumprod[t_graph][reference.batch_ligand, None]
    if not torch.isfinite(a).all() or bool(((a < 0) | (a > 1)).any()):
        raise ValueError("invalid TargetDiff alphas_cumprod")
    positions = a.sqrt() * reference.ligand_pos + (1.0 - a).sqrt() * noise
    log_probs = adapter.model.q_v_pred(
        _index_to_log_onehot(reference.ligand_v, adapter.num_classes),
        t_graph, reference.batch_ligand,
    )
    if log_probs.shape != (reference.ligand_v.numel(), adapter.num_classes) or not torch.isfinite(log_probs).all():
        raise ValueError("invalid TargetDiff categorical probabilities")
    types = _sample_log_categorical(log_probs, generator, uniform=uniform)
    return reference.replace(ligand_pos=positions, ligand_v=types)
