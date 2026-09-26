"""Teacher-forced multi-time training inputs in the fixed apo coordinate frame."""

from dataclasses import dataclass, fields
from typing import Optional

import torch

from pocketdiff.geometry.bridge import build_bridge_state, remaining_transform_current_to_holo
from pocketdiff.geometry.frames import build_residue_frames

from .clean import CleanBatch


@dataclass(frozen=True)
class BridgeBatch(CleanBatch):
    """A training batch at per-graph k, with a synthetic next-state target.

    ``protein_pos_holo`` is the experimental endpoint. ``protein_pos_next_target``
    is the residue-rigid bridge at (k+1)/20, including at k=19: internal side-chain
    deformation is outside this parameterization. Supervision fields are never
    included in the inherited ``model_kwargs``.
    """

    protein_pos_next_target: torch.Tensor

    @property
    def remaining_steps(self) -> torch.Tensor:
        """Per-graph step counts; index by batch_residue before calling the solver."""
        return 20 - self.pocket_k


@torch.no_grad()
def build_bridge_batch(
    clean: CleanBatch,
    pocket_k: Optional[torch.Tensor] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> BridgeBatch:
    """Build P_k, P_(k+1) and current-to-holo labels from clean endpoint data.

    Pass LongTensor [B] ``pocket_k`` or sample uniform k in [0, 19] using a
    generator on the input device. Explicit k and generator are mutually
    exclusive. Always start from ``collate_clean_examples`` output; do not
    recycle a previous BridgeBatch. No coordinates are re-centered, and no
    cached endpoint labels are reused as remaining labels.
    """
    if not isinstance(clean, CleanBatch) or isinstance(clean, BridgeBatch):
        raise TypeError("clean must be an endpoint CleanBatch, not a BridgeBatch")
    if pocket_k is not None and generator is not None:
        raise ValueError("provide either pocket_k or generator, not both")
    num_graphs = len(clean.sample_ids)
    device = clean.apo_pos_ref.device
    if num_graphs == 0:
        raise ValueError("clean must contain at least one graph")
    if (not torch.equal(clean.protein_pos, clean.apo_pos_ref)
            or clean.pocket_k.shape != (num_graphs,)
            or bool((clean.pocket_k != 0).any())
            or clean.targetdiff_t.shape != (num_graphs,)
            or bool((clean.targetdiff_t != 199).any())):
        raise ValueError("clean must contain apo endpoints at k=0/t=199")
    if pocket_k is not None:
        if (not isinstance(pocket_k, torch.Tensor) or pocket_k.dtype != torch.long
                or pocket_k.shape != (num_graphs,) or pocket_k.device != device):
            raise ValueError("pocket_k must be LongTensor [B] on the input device")
        if bool(((pocket_k < 0) | (pocket_k > 19)).any()):
            raise ValueError("pocket_k must lie in [0, 19]")
        k = pocket_k.clone()
    else:
        k = torch.randint(20, (num_graphs,), device=device, generator=generator)

    residue_ids = clean.atom_to_residue_global
    num_residues = clean.residue_type.numel()
    frame_kwargs = dict(atom_to_residue=residue_ids, atom_name=clean.protein_atom_name,
                        num_residues=num_residues)
    apo_frames = build_residue_frames(clean.apo_pos_ref, **frame_kwargs)
    holo_frames = build_residue_frames(clean.protein_pos_holo, **frame_kwargs)
    valid = clean.frame_valid & apo_frames.valid & holo_frames.valid
    k_residue = k[clean.batch_residue]
    geometry_kwargs = dict(
        apo_pos=clean.apo_pos_ref, atom_to_residue=residue_ids,
        apo_origins=apo_frames.origins, apo_frames=apo_frames.frames,
        holo_origins=holo_frames.origins, holo_frames=holo_frames.frames,
        frame_valid=valid,
    )
    current = build_bridge_state(fraction=k_residue.float() / 20, **geometry_kwargs)
    next_state = build_bridge_state(fraction=(k_residue.float() + 1) / 20, **geometry_kwargs)
    # Avoid round-off from R(0) construction so k=0 keeps exact clean semantics.
    current_pos = torch.where((k[clean.batch_protein] == 0)[:, None],
                              clean.apo_pos_ref, current.protein_pos)
    current_frames = build_residue_frames(current_pos, **frame_kwargs)
    # A rigid transform should never destroy a previously valid frame. Expose
    # numerical failures instead of silently dropping valid supervision.
    if bool((valid & ~current_frames.valid).any()):
        raise ValueError("bridge construction invalidated a valid residue frame")
    remaining = remaining_transform_current_to_holo(
        current_frames.origins, current_frames.frames,
        holo_frames.origins, holo_frames.frames, frame_valid=valid,
    )
    values = {field.name: getattr(clean, field.name) for field in fields(CleanBatch)}
    values.update(
        protein_pos=current_pos, frame_valid=remaining.valid,
        target_translation_local=remaining.translation_local,
        target_rotvec_local=remaining.rotvec_local,
        pocket_k=k, targetdiff_t=199 - 10 * k,
    )
    return BridgeBatch(**values, protein_pos_next_target=next_state.protein_pos)


__all__ = ["BridgeBatch", "build_bridge_batch"]
