"""Build detached training inputs from model-owned autonomous states."""

from dataclasses import dataclass, fields

import torch

from pocketdiff.geometry.bridge import remaining_transform_current_to_holo
from pocketdiff.geometry.frames import build_residue_frames

from .clean import CleanBatch
from .clean import MotionLoss, masked_remaining_motion_loss
from .bridge import BridgeBatch


@dataclass(frozen=True)
class SelfStateBatch(CleanBatch):
    """Current model state with supervision toward holo, not its old rollout.

    No next-state coordinate target is exposed: the old model's next prediction
    is an observation, not a ground-truth structure to imitate.
    """

    @property
    def remaining_steps(self):
        return 20 - self.pocket_k


@torch.no_grad()
def build_self_state_batch(
    clean: CleanBatch,
    trajectory_positions: torch.Tensor,
    pocket_k: torch.Tensor,
) -> SelfStateBatch:
    """Select per-graph autonomous states and recompute their remaining labels.

    ``trajectory_positions`` has shape ``[S, Np, 3]`` and starts at apo. A
    graph may select a different step through ``pocket_k``; atoms are selected
    by ``batch_protein``. The returned batch is fully detached and may be fed
    to the ordinary model/loss path. Holo coordinates are used only to create
    labels and are not part of ``model_kwargs``.
    """
    if type(clean) is not CleanBatch:
        raise TypeError("clean must be an endpoint CleanBatch")
    if (not torch.equal(clean.protein_pos, clean.apo_pos_ref)
            or bool((clean.pocket_k != 0).any()) or bool((clean.targetdiff_t != 199).any())):
        raise ValueError("clean must contain apo endpoints at k=0/t=199")
    if trajectory_positions.ndim != 3 or trajectory_positions.shape[1:] != clean.protein_pos.shape:
        raise ValueError("trajectory_positions must have shape [S, Np, 3]")
    if trajectory_positions.shape[0] < 21:
        raise ValueError("trajectory_positions must contain steps 0..20")
    if not trajectory_positions.is_floating_point() or not torch.isfinite(trajectory_positions).all():
        raise ValueError("trajectory_positions must be finite floating tensor")
    if pocket_k.dtype != torch.long or pocket_k.shape != clean.pocket_k.shape:
        raise ValueError("pocket_k must be LongTensor [B]")
    if bool((pocket_k < 0).any()) or bool((pocket_k > 19).any()):
        raise ValueError("pocket_k must lie in [0, 19]")
    if not torch.equal(trajectory_positions[0], clean.apo_pos_ref):
        raise ValueError("trajectory step 0 must equal clean apo coordinates")
    if trajectory_positions.device != clean.protein_pos.device or pocket_k.device != clean.pocket_k.device:
        raise ValueError("trajectory and pocket_k must use the clean batch device")

    # Select one trajectory state per graph without constructing a gradient path.
    atom_steps = pocket_k[clean.batch_protein]
    atom_index = torch.arange(clean.protein_pos.shape[0], device=clean.protein_pos.device)
    current = trajectory_positions[atom_steps, atom_index].detach().clone()
    names = clean.protein_atom_name
    kwargs = dict(atom_to_residue=clean.atom_to_residue_global, atom_name=names,
                  num_residues=clean.residue_type.numel())
    current_frames = build_residue_frames(current, **kwargs)
    holo_frames = build_residue_frames(clean.protein_pos_holo, **kwargs)
    apo_frames = build_residue_frames(clean.apo_pos_ref, **kwargs)
    valid = clean.frame_valid & current_frames.valid & apo_frames.valid & holo_frames.valid
    if bool((clean.frame_valid & ~valid).any()):
        raise ValueError("self-state invalidated a reference-valid residue frame")
    remaining = remaining_transform_current_to_holo(
        current_frames.origins, current_frames.frames,
        holo_frames.origins, holo_frames.frames, frame_valid=valid,
    )
    values = {field.name: (getattr(clean, field.name).detach()
                          if isinstance(getattr(clean, field.name), torch.Tensor)
                          else getattr(clean, field.name)) for field in fields(CleanBatch)}
    values.update(
        protein_pos=current,
        frame_valid=remaining.valid.detach(),
        target_translation_local=remaining.translation_local.detach(),
        target_rotvec_local=remaining.rotvec_local.detach(),
        targetdiff_t=(199 - 10 * pocket_k).detach().clone(),
        pocket_k=pocket_k.detach().clone(),
    )
    # Preserve endpoint metadata and force all newly constructed tensors out of
    # any caller graph. Existing clean endpoint tensors are already immutable by
    # contract; clone the state-dependent fields above.
    return SelfStateBatch(**values)


def masked_self_state_motion_loss(
    prediction,
    target_translation_local: torch.Tensor,
    target_rotvec_local: torch.Tensor,
    target_valid: torch.Tensor,
) -> MotionLoss:
    """Physical-unit loss for autonomous self-state labels.

    This deliberately delegates to the original remaining-motion loss: no
    division by the ideal bridge rate, temporal reweighting, clipping or
    rotation-bound remapping is applied. The model output is therefore the
    actual current-state-to-holo transform.
    """
    return masked_remaining_motion_loss(prediction, target_translation_local,
                                        target_rotvec_local, target_valid)


__all__ = ["SelfStateBatch", "build_self_state_batch", "masked_self_state_motion_loss"]
