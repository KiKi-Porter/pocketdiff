"""One auditable PocketDiff -> TargetDiff ten-step coupling block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from pocketdiff.data.schema import PocketStepOutput, ResidueMetadata
from pocketdiff.targetdiff.adapter import (
    TargetDiffAdapter,
    TargetDiffRNGTrace,
    TargetDiffStepAux,
)
from pocketdiff.targetdiff.state import TargetDiffState

from .pocket_solver import PocketStepSolver


@dataclass(frozen=True)
class PocketBlockEvent:
    """Machine-readable schedule and invariant facts for one coupling block."""

    k: int
    pocket_targetdiff_t: int
    targetdiff_timesteps: Tuple[int, ...]
    protein_changed_by_pocket: bool
    ligand_unchanged_during_pocket: bool
    protein_features_unchanged: bool
    apo_reference_unchanged: bool
    batch_unchanged: bool
    center_offset_unchanged: bool
    protein_checksum_before: float
    protein_checksum_after_pocket: float
    protein_checksum_after_block: float
    ligand_checksum_before: float
    ligand_checksum_after_pocket: float
    ligand_checksum_after_block: float

    def __post_init__(self) -> None:
        if self.k < 0 or self.k > 19:
            raise ValueError("k must lie in [0, 19]")
        expected_t = 199 - 10 * self.k
        if self.pocket_targetdiff_t != expected_t:
            raise ValueError("pocket_targetdiff_t is inconsistent with k")
        expected_sequence = tuple(range(expected_t, expected_t - 10, -1))
        if self.targetdiff_timesteps != expected_sequence:
            raise ValueError("targetdiff_timesteps must contain the descending ten-step block")


@dataclass(frozen=True)
class PocketBlockResult:
    """State and diagnostics returned by :func:`run_pocket_block`."""

    state: TargetDiffState
    pocket_output: PocketStepOutput
    targetdiff_aux: Tuple[TargetDiffStepAux, ...]
    event: PocketBlockEvent

    def __post_init__(self) -> None:
        if len(self.targetdiff_aux) != len(self.event.targetdiff_timesteps):
            raise ValueError("targetdiff_aux length does not match the block schedule")


def _scalar_k(k: Union[int, torch.Tensor]) -> int:
    if isinstance(k, bool):
        raise TypeError("k must be an integer or a scalar LongTensor")
    if isinstance(k, int):
        value = k
    elif isinstance(k, torch.Tensor):
        if k.dtype != torch.long or k.ndim != 0:
            raise ValueError("k must be an integer or a scalar LongTensor")
        value = int(k.detach().item())
    else:
        raise TypeError("k must be an integer or a scalar LongTensor")
    if value < 0 or value > 19:
        raise ValueError("k must lie in [0, 19]")
    return value


def _checksum(value: torch.Tensor) -> float:
    return float(value.detach().double().sum().item())


@torch.no_grad()
def run_pocket_block(
    adapter: TargetDiffAdapter,
    pocket_solver: PocketStepSolver,
    state: TargetDiffState,
    residue_metadata: ResidueMetadata,
    k: Union[int, torch.Tensor],
    *,
    generator: Optional[torch.Generator] = None,
    rng_trace: Optional[TargetDiffRNGTrace] = None,
) -> PocketBlockResult:
    """Apply one PocketDiff step followed by ten TargetDiff reverse steps.

    The block intentionally accepts only a scalar ``k``.  All graphs in a
    batch therefore share the same TargetDiff timestep; asynchronous graph
    schedules require a separate adapter contract and are deferred.
    """

    if not isinstance(adapter, TargetDiffAdapter):
        raise TypeError("adapter must be a TargetDiffAdapter")
    if not isinstance(pocket_solver, PocketStepSolver):
        raise TypeError("pocket_solver must be a PocketStepSolver")
    if not isinstance(state, TargetDiffState):
        raise TypeError("state must be a TargetDiffState")
    if not isinstance(residue_metadata, ResidueMetadata):
        raise TypeError("residue_metadata must be a ResidueMetadata")
    if generator is not None and rng_trace is not None:
        raise ValueError("provide either generator or rng_trace, not both")
    k_value = _scalar_k(k)
    pocket_t = 199 - 10 * k_value
    timesteps = tuple(range(pocket_t, pocket_t - 10, -1))

    protein_before = state.protein_pos.clone()
    ligand_pos_before = state.ligand_pos.clone()
    ligand_v_before = state.ligand_v.clone()
    protein_feature_before = state.protein_v.clone()
    apo_before = state.apo_pos_ref.clone()
    batch_protein_before = state.batch_protein.clone()
    batch_ligand_before = state.batch_ligand.clone()
    center_offset_before = state.center_offset.clone()

    pocket_output = pocket_solver.step(state, residue_metadata, k_value)
    state_after_pocket = state.replace(protein_pos=pocket_output.protein_pos_next)
    ligand_unchanged_during_pocket = bool(
        torch.equal(state_after_pocket.ligand_pos, ligand_pos_before)
        and torch.equal(state_after_pocket.ligand_v, ligand_v_before)
    )

    next_state, auxiliaries = adapter.run_steps(
        state_after_pocket,
        t_start=timesteps[0],
        t_end_inclusive=timesteps[-1],
        generator=generator,
        rng_trace=rng_trace,
    )
    event = PocketBlockEvent(
        k=k_value,
        pocket_targetdiff_t=pocket_t,
        targetdiff_timesteps=timesteps,
        protein_changed_by_pocket=bool(not torch.equal(pocket_output.protein_pos_next, protein_before)),
        ligand_unchanged_during_pocket=ligand_unchanged_during_pocket,
        protein_features_unchanged=bool(torch.equal(next_state.protein_v, protein_feature_before)),
        apo_reference_unchanged=bool(torch.equal(next_state.apo_pos_ref, apo_before)),
        batch_unchanged=bool(
            torch.equal(next_state.batch_protein, batch_protein_before)
            and torch.equal(next_state.batch_ligand, batch_ligand_before)
        ),
        center_offset_unchanged=bool(torch.equal(next_state.center_offset, center_offset_before)),
        protein_checksum_before=_checksum(protein_before),
        protein_checksum_after_pocket=_checksum(pocket_output.protein_pos_next),
        protein_checksum_after_block=_checksum(next_state.protein_pos),
        ligand_checksum_before=_checksum(ligand_pos_before),
        ligand_checksum_after_pocket=_checksum(state_after_pocket.ligand_pos),
        ligand_checksum_after_block=_checksum(next_state.ligand_pos),
    )
    return PocketBlockResult(
        state=next_state,
        pocket_output=pocket_output,
        targetdiff_aux=tuple(auxiliaries),
        event=event,
    )


__all__ = ["PocketBlockEvent", "PocketBlockResult", "run_pocket_block"]
