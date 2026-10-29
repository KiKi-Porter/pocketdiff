"""Tensor state contract used by the minimal TargetDiff adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


def _require_tensor(name: str, value: torch.Tensor, ndim: int, *, dtype=None, last_shape=None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(value.shape)}")
    if last_shape is not None and tuple(value.shape[-len(last_shape):]) != tuple(last_shape):
        raise ValueError(f"{name} must end with shape {tuple(last_shape)}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.numel() and (value.is_floating_point() and not torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_batch(name: str, batch: torch.Tensor, count: int) -> None:
    _require_tensor(name, batch, 1, dtype=torch.long)
    if batch.numel() != count or count == 0:
        raise ValueError(f"{name} must contain one graph id per item")
    if int(batch.min()) < 0:
        raise ValueError(f"{name} cannot contain negative graph ids")


def _require_floating(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")


@dataclass(frozen=True)
class TargetDiffState:
    """One centered TargetDiff state; only ligand tensors change per step."""

    protein_pos: torch.Tensor
    protein_v: torch.Tensor
    batch_protein: torch.Tensor
    ligand_pos: torch.Tensor
    ligand_v: torch.Tensor
    batch_ligand: torch.Tensor
    apo_pos_ref: torch.Tensor
    center_offset: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor("protein_pos", self.protein_pos, 2, last_shape=(3,))
        _require_tensor("protein_v", self.protein_v, 2, last_shape=(27,))
        _require_tensor("ligand_pos", self.ligand_pos, 2, last_shape=(3,))
        _require_tensor("ligand_v", self.ligand_v, 1, dtype=torch.long)
        _require_tensor("apo_pos_ref", self.apo_pos_ref, 2, last_shape=(3,))
        _require_tensor("center_offset", self.center_offset, 2, last_shape=(3,))
        for name, value in (
            ("protein_pos", self.protein_pos),
            ("protein_v", self.protein_v),
            ("ligand_pos", self.ligand_pos),
            ("apo_pos_ref", self.apo_pos_ref),
            ("center_offset", self.center_offset),
        ):
            _require_floating(name, value)
        if self.protein_pos.shape != self.apo_pos_ref.shape:
            raise ValueError("protein_pos and apo_pos_ref must have identical shape")
        if self.protein_v.shape[0] != self.protein_pos.shape[0]:
            raise ValueError("protein_v and protein_pos have different atom counts")
        if self.ligand_v.shape[0] != self.ligand_pos.shape[0] or self.ligand_pos.shape[0] == 0:
            raise ValueError("ligand tensors must have the same non-zero atom count")
        if self.ligand_v.numel() and (int(self.ligand_v.min()) < 0 or int(self.ligand_v.max()) >= 13):
            raise ValueError("ligand_v must lie in [0, 12]")
        _validate_batch("batch_protein", self.batch_protein, self.protein_pos.shape[0])
        _validate_batch("batch_ligand", self.batch_ligand, self.ligand_pos.shape[0])
        graph_count = int(self.batch_protein.max().item()) + 1
        if int(self.batch_ligand.max().item()) >= graph_count:
            raise ValueError("batch_ligand contains an out-of-range graph id")
        if self.center_offset.shape[0] != graph_count:
            raise ValueError("center_offset must have one row per graph")

    @property
    def num_graphs(self) -> int:
        return int(self.batch_protein.max().item()) + 1

    def replace(self, **changes) -> "TargetDiffState":
        """Return a new state without mutating any existing tensor."""

        return replace(self, **changes)


def initialize_targetdiff_state(
    *,
    protein_pos: torch.Tensor,
    protein_v: torch.Tensor,
    batch_protein: torch.Tensor,
    ligand_pos: torch.Tensor,
    ligand_v: torch.Tensor,
    batch_ligand: torch.Tensor,
    apo_pos_ref: Optional[torch.Tensor] = None,
    center_mode: str = "protein",
) -> TargetDiffState:
    """Create a centered state using the official ``center_pos`` convention."""

    if center_mode not in ("protein", "none"):
        raise ValueError("center_mode must be 'protein' or 'none'")
    _require_tensor("protein_pos", protein_pos, 2, last_shape=(3,))
    _require_tensor("protein_v", protein_v, 2, last_shape=(27,))
    _require_tensor("ligand_pos", ligand_pos, 2, last_shape=(3,))
    _require_tensor("ligand_v", ligand_v, 1, dtype=torch.long)
    for name, value in (("protein_pos", protein_pos), ("protein_v", protein_v), ("ligand_pos", ligand_pos)):
        _require_floating(name, value)
    _validate_batch("batch_protein", batch_protein, protein_pos.shape[0])
    _validate_batch("batch_ligand", batch_ligand, ligand_pos.shape[0])
    if protein_v.shape[0] != protein_pos.shape[0]:
        raise ValueError("protein_v and protein_pos have different atom counts")
    if ligand_v.shape[0] != ligand_pos.shape[0]:
        raise ValueError("ligand_v and ligand_pos have different atom counts")
    if apo_pos_ref is None:
        apo_pos_ref = protein_pos
    _require_tensor("apo_pos_ref", apo_pos_ref, 2, last_shape=(3,))
    _require_floating("apo_pos_ref", apo_pos_ref)
    if apo_pos_ref.shape != protein_pos.shape:
        raise ValueError("apo_pos_ref and protein_pos must have identical shape")
    graph_count = int(batch_protein.max().item()) + 1
    if batch_ligand.numel() and int(batch_ligand.max().item()) >= graph_count:
        raise ValueError("batch_ligand contains an out-of-range graph id")
    if center_mode == "protein":
        offset = torch.zeros((graph_count, 3), dtype=protein_pos.dtype, device=protein_pos.device)
        offset.index_add_(0, batch_protein, protein_pos)
        counts = torch.bincount(batch_protein, minlength=graph_count).to(dtype=protein_pos.dtype)
        offset = offset / counts.clamp_min(1.0)[:, None]
        centered_protein = protein_pos - offset[batch_protein]
        centered_ligand = ligand_pos - offset[batch_ligand]
        centered_apo = apo_pos_ref - offset[batch_protein]
    else:
        offset = torch.zeros((graph_count, 3), dtype=protein_pos.dtype, device=protein_pos.device)
        centered_protein, centered_ligand, centered_apo = protein_pos, ligand_pos, apo_pos_ref
    return TargetDiffState(
        protein_pos=centered_protein.clone(),
        protein_v=protein_v.clone(),
        batch_protein=batch_protein.clone(),
        ligand_pos=centered_ligand.clone(),
        ligand_v=ligand_v.clone(),
        batch_ligand=batch_ligand.clone(),
        apo_pos_ref=centered_apo.clone(),
        center_offset=offset.clone(),
    )


def restore_center(value: torch.Tensor, batch: torch.Tensor, center_offset: torch.Tensor) -> torch.Tensor:
    """Add the initialization offset back to centered coordinates."""

    _require_tensor("value", value, 2, last_shape=(3,))
    _require_floating("value", value)
    _validate_batch("batch", batch, value.shape[0])
    _require_tensor("center_offset", center_offset, 2, last_shape=(3,))
    _require_floating("center_offset", center_offset)
    if int(batch.max().item()) >= center_offset.shape[0]:
        raise ValueError("batch references a missing center offset")
    return value + center_offset[batch]


__all__ = ["TargetDiffState", "initialize_targetdiff_state", "restore_center"]
