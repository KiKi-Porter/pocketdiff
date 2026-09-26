"""Coordinate-level losses for independent PocketDiff one-step training."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class NextXYZLoss:
    loss: torch.Tensor
    graph_losses: torch.Tensor
    valid_atom_count: int
    valid_graph_count: int


def masked_next_xyz_loss(predicted_pos: torch.Tensor, target_pos: torch.Tensor,
                         atom_to_residue: torch.Tensor, frame_valid: torch.Tensor,
                         batch_protein: torch.Tensor) -> NextXYZLoss:
    """Compute graph-balanced mean valid-atom squared coordinate error.

    The reduction is ``mean_graph(mean_valid_atoms(per_atom_squared_error))``;
    graphs with no valid residue atoms are excluded explicitly. Coordinates are
    compared in the fixed apo frame, with no Kabsch realignment.
    """
    if predicted_pos.ndim != 2 or predicted_pos.shape[-1] != 3 or predicted_pos.shape != target_pos.shape:
        raise ValueError('predicted_pos and target_pos must both have shape [N,3]')
    if not predicted_pos.is_floating_point() or not target_pos.is_floating_point():
        raise TypeError('coordinate tensors must be floating')
    if not (torch.isfinite(predicted_pos).all() and torch.isfinite(target_pos).all()):
        raise ValueError('coordinate tensors must be finite')
    if atom_to_residue.dtype != torch.long or atom_to_residue.shape != (predicted_pos.shape[0],):
        raise ValueError('atom_to_residue must be LongTensor [N]')
    if frame_valid.dtype != torch.bool or frame_valid.ndim != 1:
        raise ValueError('frame_valid must be BoolTensor [Nr]')
    if batch_protein.dtype != torch.long or batch_protein.shape != (predicted_pos.shape[0],):
        raise ValueError('batch_protein must be LongTensor [N]')
    if any(v.device != predicted_pos.device for v in (target_pos, atom_to_residue, frame_valid, batch_protein)):
        raise ValueError('coordinate loss tensors must share a device')
    if atom_to_residue.numel() and (int(atom_to_residue.min()) < 0 or int(atom_to_residue.max()) >= frame_valid.numel()):
        raise ValueError('atom_to_residue contains an invalid residue')
    if batch_protein.numel() == 0 or int(batch_protein.min()) < 0:
        raise ValueError('batch_protein must contain nonnegative graph ids')
    per_atom = (predicted_pos - target_pos).square().mean(dim=-1)
    valid_atom = frame_valid[atom_to_residue]
    graph_count = int(batch_protein.max()) + 1
    graph_losses = []
    for graph_id in range(graph_count):
        selected = valid_atom & (batch_protein == graph_id)
        if selected.any():
            graph_losses.append(per_atom[selected].mean())
    if not graph_losses:
        raise ValueError('coordinate loss has no valid graph')
    graph_losses = torch.stack(graph_losses)
    return NextXYZLoss(graph_losses.mean(), graph_losses, int(valid_atom.sum()), int(graph_losses.numel()))


__all__ = ['NextXYZLoss', 'masked_next_xyz_loss']
