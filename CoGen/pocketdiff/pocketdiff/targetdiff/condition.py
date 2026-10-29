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
        if self.targetdiff_t < -1:
            raise ValueError("targetdiff_t must be -1 (clean) or a valid timestep")


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
        return self._condition_at_t(
            protein_pos=protein_pos,
            protein_v=protein_feature,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_type,
            batch_ligand=batch_ligand,
            targetdiff_t=self._targetdiff_time(pocket_time),
            generator=generator,
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
        return self._condition_at_t(
            protein_pos=protein_pos,
            protein_v=protein_feature,
            batch_protein=batch_protein.to(device=self.adapter.device),
            ligand_pos=ligand_pos,
            ligand_v=ligand_type,
            batch_ligand=batch_ligand.to(device=self.adapter.device),
            targetdiff_t=self._targetdiff_time(pocket_time),
            generator=generator,
        )

    def _condition_at_t(
        self,
        *,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_protein: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        targetdiff_t: int,
        generator: Optional[torch.Generator],
    ) -> TargetDiffLigandCondition:
        reference = initialize_targetdiff_state(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            apo_pos_ref=protein_pos,
            center_mode="protein",
        )
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


class TargetDiffTrainingLigandConditionProvider(TargetDiffLigandConditionProvider):
    """Sample clean or low-noise real ligand conditions for PocketDiff training.

    The noisy branch uses TargetDiff's forward kernels directly from each
    clean Apo2Mol ligand. It never calls TargetDiff's learned reverse model.
    A condition already attached to a model input is reused through the whole
    PocketDiff rollout so the ligand does not randomly jump between steps.
    """

    def __init__(
        self,
        adapter: TargetDiffAdapter,
        *,
        t_min: int = 0,
        t_max: int = 199,
        clean_probability: float = 0.20,
    ):
        super().__init__(adapter)
        if not isinstance(t_min, int) or isinstance(t_min, bool) or t_min < 0:
            raise ValueError("t_min must be a non-negative integer")
        if (
            not isinstance(t_max, int)
            or isinstance(t_max, bool)
            or t_max < t_min
            or t_max >= adapter.num_timesteps
        ):
            raise ValueError("t_max must lie in [t_min, num_timesteps)")
        if not 0.0 <= float(clean_probability) <= 1.0:
            raise ValueError("clean_probability must lie in [0, 1]")
        self.t_min = t_min
        self.t_max = t_max
        self.clean_probability = float(clean_probability)

    def _draw_timestep(
        self,
        generator: Optional[torch.Generator],
    ) -> Optional[int]:
        draw = torch.rand((), device=self.adapter.device, generator=generator)
        if float(draw) < self.clean_probability:
            return None
        return int(
            torch.randint(
                self.t_min,
                self.t_max + 1,
                (),
                dtype=torch.long,
                device=self.adapter.device,
                generator=generator,
            ).item()
        )

    @staticmethod
    def _validate_pocket_time(pocket_time: float) -> None:
        if (
            not isinstance(pocket_time, (float, int))
            or isinstance(pocket_time, bool)
            or not 0.0 <= float(pocket_time) <= 1.0
        ):
            raise ValueError("pocket_time must lie in [0, 1]")

    def _sample_condition(
        self,
        *,
        protein_pos: torch.Tensor,
        protein_feature: torch.Tensor,
        batch_protein: torch.Tensor,
        ligand_pos_ref: torch.Tensor,
        ligand_type_ref: torch.Tensor,
        batch_ligand: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> TargetDiffLigandCondition:
        timestep = self._draw_timestep(generator)
        if timestep is None:
            return TargetDiffLigandCondition(
                ligand_pos=ligand_pos_ref.detach().clone(),
                ligand_type=ligand_type_ref.detach().clone(),
                targetdiff_t=-1,
            )
        return self._condition_at_t(
            protein_pos=protein_pos,
            protein_v=protein_feature,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos_ref,
            ligand_v=ligand_type_ref,
            batch_ligand=batch_ligand,
            targetdiff_t=timestep,
            generator=generator,
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
        self._validate_pocket_time(pocket_time)
        device = self.adapter.device
        protein_pos = complex_value.protein_pos_apo.to(device=device, dtype=torch.float32)
        ligand_pos_ref = complex_value.ligand_pos_ref.to(
            device=device, dtype=torch.float32
        )
        return self._sample_condition(
            protein_pos=protein_pos,
            protein_feature=complex_value.protein_feature.to(
                device=device, dtype=torch.float32
            ),
            batch_protein=torch.zeros(
                protein_pos.shape[0], dtype=torch.long, device=device
            ),
            ligand_pos_ref=ligand_pos_ref,
            ligand_type_ref=complex_value.ligand_type_ref.to(
                device=device, dtype=torch.long
            ),
            batch_ligand=torch.zeros(
                ligand_pos_ref.shape[0], dtype=torch.long, device=device
            ),
            generator=generator,
        )

    def condition_from_input(
        self,
        model_input,
        pocket_time: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> TargetDiffLigandCondition:
        self._validate_pocket_time(pocket_time)
        if model_input.ligand_pos_ref is None or model_input.ligand_type_ref is None:
            raise ValueError("model_input must carry clean ligand reference tensors")
        if model_input.ligand_condition_t is not None:
            time = model_input.ligand_condition_t.reshape(-1)
            if time.numel() != 1:
                raise ValueError(
                    "training ligand condition reuse currently requires one graph"
                )
            return TargetDiffLigandCondition(
                ligand_pos=model_input.ligand_pos.detach().clone(),
                ligand_type=model_input.ligand_type.detach().clone(),
                targetdiff_t=int(time[0]),
            )
        batch_protein = model_input.batch_protein
        batch_ligand = model_input.batch_ligand
        if batch_protein is None:
            batch_protein = torch.zeros(
                model_input.protein_pos.shape[0],
                dtype=torch.long,
                device=self.adapter.device,
            )
        if batch_ligand is None:
            batch_ligand = torch.zeros(
                model_input.ligand_pos_ref.shape[0],
                dtype=torch.long,
                device=self.adapter.device,
            )
        graph_ids = torch.unique(torch.cat((batch_protein, batch_ligand)), sorted=True)
        if graph_ids.numel() != 1 or int(graph_ids[0]) != 0:
            raise ValueError(
                "condition_from_input currently requires one contiguous graph"
            )
        protein_pos = (
            model_input.apo_pos_ref
            if model_input.apo_pos_ref is not None
            else model_input.protein_pos
        ).to(device=self.adapter.device, dtype=torch.float32)
        return self._sample_condition(
            protein_pos=protein_pos,
            protein_feature=model_input.protein_feature.to(
                device=self.adapter.device, dtype=torch.float32
            ),
            batch_protein=batch_protein.to(device=self.adapter.device),
            ligand_pos_ref=model_input.ligand_pos_ref.to(
                device=self.adapter.device, dtype=torch.float32
            ),
            ligand_type_ref=model_input.ligand_type_ref.to(
                device=self.adapter.device, dtype=torch.long
            ),
            batch_ligand=batch_ligand.to(device=self.adapter.device),
            generator=generator,
        )


__all__ = [
    "TargetDiffLigandCondition",
    "TargetDiffLigandConditionProvider",
    "TargetDiffTrainingLigandConditionProvider",
]
