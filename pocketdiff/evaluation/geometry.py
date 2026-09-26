"""Read-only geometry diagnostics for apo-to-holo protein motion."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProteinMotionDiagnostics:
    """Scalar observations for one centered final protein state."""

    apo_to_holo_rmsd: float
    final_to_holo_rmsd: float
    rmsd_improvement: float
    max_final_displacement: float
    mean_final_displacement: float
    num_atoms: int

    def as_dict(self):
        return {
            "apo_to_holo_rmsd": self.apo_to_holo_rmsd,
            "final_to_holo_rmsd": self.final_to_holo_rmsd,
            "rmsd_improvement": self.rmsd_improvement,
            "max_final_displacement": self.max_final_displacement,
            "mean_final_displacement": self.mean_final_displacement,
            "num_atoms": self.num_atoms,
        }


def _require_coordinates(name: str, value: torch.Tensor, shape):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [Np, 3]")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")


def evaluate_protein_motion(
    protein_pos_apo: torch.Tensor,
    protein_pos_holo: torch.Tensor,
    final_protein_pos_centered: torch.Tensor,
    center_offset: torch.Tensor,
    batch_protein: torch.Tensor,
) -> ProteinMotionDiagnostics:
    """Compare a centered final protein with raw apo/holo references.

    ``center_offset`` must be the exact offset used when initializing the
    TargetDiff state.  Apo and holo are centered with this same offset; the
    final state is already centered and is never recentered here.
    """

    if not isinstance(batch_protein, torch.Tensor) or batch_protein.dtype != torch.long or batch_protein.ndim != 1:
        raise ValueError("batch_protein must be a LongTensor [Np]")
    _require_coordinates("protein_pos_apo", protein_pos_apo, protein_pos_apo.shape)
    _require_coordinates("protein_pos_holo", protein_pos_holo, protein_pos_apo.shape)
    _require_coordinates("final_protein_pos_centered", final_protein_pos_centered, protein_pos_apo.shape)
    if batch_protein.shape[0] != protein_pos_apo.shape[0] or not batch_protein.numel():
        raise ValueError("batch_protein must contain one graph id per protein atom")
    if int(batch_protein.min()) < 0:
        raise ValueError("batch_protein cannot contain negative graph ids")
    if not isinstance(center_offset, torch.Tensor) or center_offset.ndim != 2 or center_offset.shape[-1] != 3:
        raise ValueError("center_offset must have shape [B, 3]")
    if not center_offset.is_floating_point() or not torch.isfinite(center_offset).all():
        raise ValueError("center_offset must be finite floating point")
    num_graphs = int(batch_protein.max().item()) + 1
    if center_offset.shape[0] != num_graphs:
        raise ValueError("center_offset must contain one row per graph")
    if not (protein_pos_apo.device == protein_pos_holo.device == final_protein_pos_centered.device == center_offset.device == batch_protein.device):
        raise ValueError("all geometry tensors must be on the same device")

    offset_per_atom = center_offset[batch_protein]
    apo_centered = protein_pos_apo - offset_per_atom
    holo_centered = protein_pos_holo - offset_per_atom
    apo_to_holo_error = torch.linalg.vector_norm(apo_centered - holo_centered, dim=-1)
    final_to_holo_error = torch.linalg.vector_norm(final_protein_pos_centered - holo_centered, dim=-1)
    final_displacement = torch.linalg.vector_norm(final_protein_pos_centered - apo_centered, dim=-1)
    apo_rmsd = torch.sqrt((apo_to_holo_error.square()).mean())
    final_rmsd = torch.sqrt((final_to_holo_error.square()).mean())
    return ProteinMotionDiagnostics(
        apo_to_holo_rmsd=float(apo_rmsd.item()),
        final_to_holo_rmsd=float(final_rmsd.item()),
        rmsd_improvement=float((apo_rmsd - final_rmsd).item()),
        max_final_displacement=float(final_displacement.max().item()),
        mean_final_displacement=float(final_displacement.mean().item()),
        num_atoms=int(protein_pos_apo.shape[0]),
    )


__all__ = ["ProteinMotionDiagnostics", "evaluate_protein_motion"]
