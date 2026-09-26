"""Optional TargetDiff forward-noised ligand conditions for PocketDiff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from pocketdiff.data.schema import PocketComplex

from .adapter import TargetDiffAdapter
from .forward_noise import forward_noise_reference
from .state import initialize_targetdiff_state, restore_center


@dataclass(frozen=True)
class TargetDiffLigandCondition:
    """A ligand condition in the PocketDiff coordinate frame."""

    ligand_pos: torch.Tensor
    ligand_type: torch.Tensor
    targetdiff_t: int

    def __post_init__(self) -> None:
        if self.ligand_pos.ndim != 2 or self.ligand_pos.shape[-1] != 3:
            raise ValueError("ligand_pos must have shape [Nl, 3]")
        if self.ligand_type.dtype != torch.long or self.ligand_type.shape != (
            self.ligand_pos.shape[0],
        ):
            raise ValueError("ligand_type must be LongTensor [Nl]")
        if not torch.isfinite(self.ligand_pos).all():
            raise ValueError("ligand_pos contains non-finite values")
        if not isinstance(self.targetdiff_t, int) or isinstance(self.targetdiff_t, bool):
            raise TypeError("targetdiff_t must be an integer")
        if self.targetdiff_t < 0:
            raise ValueError("targetdiff_t must be non-negative")


class TargetDiffLigandConditionProvider:
    """Build independent q(L_t | clean reference) conditions.

    The official TargetDiff score model is never called.  Only its loaded
    schedule and categorical forward kernel are used, so each condition is
    reproducible and cannot accidentally become a learned ligand rollout.
    """

    def __init__(self, adapter: TargetDiffAdapter):
        if not isinstance(adapter, TargetDiffAdapter):
            raise TypeError("adapter must be a TargetDiffAdapter")
        self.adapter = adapter

    def _targetdiff_time(self, pocket_time: float) -> int:
        if not isinstance(pocket_time, (float, int)) or not 0.0 <= float(pocket_time) <= 1.0:
            raise ValueError("pocket_time must lie in [0, 1]")
        return max(
            0,
            min(
                self.adapter.num_timesteps - 1,
                int(round(float(pocket_time) * (self.adapter.num_timesteps - 1))),
            ),
        )

    def condition(
        self,
        complex_value: PocketComplex,
        pocket_time: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> TargetDiffLigandCondition:
        if not isinstance(complex_value, PocketComplex):
            raise TypeError("complex_value must be PocketComplex")
        device = self.adapter.device
        protein_pos = complex_value.protein_pos_apo.to(device=device, dtype=torch.float32)
        protein_feature = complex_value.protein_feature.to(device=device, dtype=torch.float32)
        ligand_pos = complex_value.ligand_pos_ref.to(device=device, dtype=torch.float32)
        ligand_type = complex_value.ligand_type_ref.to(device=device, dtype=torch.long)
        batch_protein = torch.zeros(
            protein_pos.shape[0], dtype=torch.long, device=device
        )
        batch_ligand = torch.zeros(
            ligand_pos.shape[0], dtype=torch.long, device=device
        )
        reference = initialize_targetdiff_state(
            protein_pos=protein_pos,
            protein_v=protein_feature,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_type,
            batch_ligand=batch_ligand,
            apo_pos_ref=protein_pos,
            center_mode="protein",
        )
        targetdiff_t = self._targetdiff_time(pocket_time)
        noised = forward_noise_reference(
            self.adapter,
            reference,
            torch.tensor([targetdiff_t], dtype=torch.long, device=device),
            generator=generator,
        )
        restored_pos = restore_center(
            noised.ligand_pos,
            noised.batch_ligand,
            noised.center_offset,
        )
        return TargetDiffLigandCondition(
            ligand_pos=restored_pos.detach().clone(),
            ligand_type=noised.ligand_v.detach().clone(),
            targetdiff_t=targetdiff_t,
        )

    def condition_from_input(self, model_input, pocket_time: float, *, generator=None):
        if model_input.ligand_pos_ref is None or model_input.ligand_type_ref is None:
            raise ValueError("model_input must carry clean ligand reference tensors")
        batch_protein = model_input.batch_protein
        batch_ligand = model_input.batch_ligand
        if batch_protein is None or batch_ligand is None:
            batch_protein = torch.zeros(
                model_input.protein_pos.shape[0],
                dtype=torch.long,
                device=self.adapter.device,
            )
            batch_ligand = torch.zeros(
                model_input.ligand_pos_ref.shape[0],
                dtype=torch.long,
                device=self.adapter.device,
            )
        graph_ids = torch.unique(
            torch.cat((batch_protein, batch_ligand)), sorted=True
        )
        if graph_ids.numel() != 1 or int(graph_ids[0]) != 0:
            raise ValueError(
                "condition_from_input currently requires one contiguous graph"
            )
        protein_pos = (
            model_input.apo_pos_ref
            if model_input.apo_pos_ref is not None
            else model_input.protein_pos
        ).to(device=self.adapter.device, dtype=torch.float32)
        protein_feature = model_input.protein_feature.to(
            device=self.adapter.device, dtype=torch.float32
        )
        ligand_pos = model_input.ligand_pos_ref.to(
            device=self.adapter.device, dtype=torch.float32
        )
        ligand_type = model_input.ligand_type_ref.to(
            device=self.adapter.device, dtype=torch.long
        )
        reference = initialize_targetdiff_state(
            protein_pos=protein_pos,
            protein_v=protein_feature,
            batch_protein=batch_protein.to(device=self.adapter.device),
            ligand_pos=ligand_pos,
            ligand_v=ligand_type,
            batch_ligand=batch_ligand.to(device=self.adapter.device),
            apo_pos_ref=protein_pos,
            center_mode="protein",
        )
        targetdiff_t = self._targetdiff_time(pocket_time)
        noised = forward_noise_reference(
            self.adapter,
            reference,
            torch.tensor([targetdiff_t], dtype=torch.long, device=self.adapter.device),
            generator=generator,
        )
        restored_pos = restore_center(
            noised.ligand_pos,
            noised.batch_ligand,
            noised.center_offset,
        )
        return TargetDiffLigandCondition(
            ligand_pos=restored_pos.detach().clone(),
            ligand_type=noised.ligand_v.detach().clone(),
            targetdiff_t=targetdiff_t,
        )


__all__ = ["TargetDiffLigandCondition", "TargetDiffLigandConditionProvider"]
