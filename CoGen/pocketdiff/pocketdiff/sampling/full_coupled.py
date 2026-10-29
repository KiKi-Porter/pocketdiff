"""Complete v0.1 TargetDiff/PocketDiff schedule built from verified blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from pocketdiff.data.schema import ResidueMetadata
from pocketdiff.targetdiff.adapter import TargetDiffAdapter, TargetDiffRNGTrace, TargetDiffStepAux
from pocketdiff.targetdiff.state import TargetDiffState

from .coupled_block import PocketBlockResult, run_pocket_block
from .pocket_solver import PocketStepSolver


@dataclass(frozen=True)
class CoupledSamplingEvent:
    """Counts, schedules, and state invariants for a full coupled run."""

    prelude_timesteps: Tuple[int, ...]
    pocket_timesteps: Tuple[int, ...]
    block_targetdiff_timesteps: Tuple[int, ...]
    targetdiff_timesteps: Tuple[int, ...]
    targetdiff_call_count: int
    pocket_call_count: int
    prelude_protein_unchanged: bool
    protein_features_unchanged: bool
    apo_reference_unchanged: bool
    batch_unchanged: bool
    center_offset_unchanged: bool
    ligand_types_valid: bool

    def __post_init__(self) -> None:
        expected_prelude = tuple(range(999, 199, -1))
        expected_pocket = tuple(199 - 10 * k for k in range(20))
        expected_blocks = tuple(
            timestep
            for pocket_t in expected_pocket
            for timestep in range(pocket_t, pocket_t - 10, -1)
        )
        expected_all = expected_prelude + expected_blocks
        if self.prelude_timesteps != expected_prelude:
            raise ValueError("prelude timesteps must be 999 down to 200")
        if self.pocket_timesteps != expected_pocket:
            raise ValueError("pocket timesteps must be [199, 189, ..., 9]")
        if self.block_targetdiff_timesteps != expected_blocks:
            raise ValueError("block TargetDiff timesteps are inconsistent")
        if self.targetdiff_timesteps != expected_all:
            raise ValueError("full TargetDiff schedule must be 999 down to 0")
        if self.targetdiff_call_count != 1000 or self.pocket_call_count != 20:
            raise ValueError("full coupled schedule must contain 1000 TargetDiff and 20 PocketDiff calls")


@dataclass(frozen=True)
class CoupledSamplingResult:
    """Final state plus every staged result from a full coupled run."""

    state: TargetDiffState
    prelude_aux: Tuple[TargetDiffStepAux, ...]
    blocks: Tuple[PocketBlockResult, ...]
    event: CoupledSamplingEvent

    def __post_init__(self) -> None:
        if len(self.prelude_aux) != 800:
            raise ValueError("prelude_aux must contain 800 TargetDiff steps")
        if len(self.blocks) != 20:
            raise ValueError("blocks must contain 20 PocketDiff blocks")


def _clone_state_tensors(state: TargetDiffState):
    return {
        name: getattr(state, name).clone()
        for name in (
            "protein_pos",
            "protein_v",
            "batch_protein",
            "ligand_pos",
            "ligand_v",
            "batch_ligand",
            "apo_pos_ref",
            "center_offset",
        )
    }


@torch.no_grad()
def run_coupled_sampling(
    adapter: TargetDiffAdapter,
    pocket_solver: PocketStepSolver,
    state: TargetDiffState,
    residue_metadata: ResidueMetadata,
    *,
    generator: Optional[torch.Generator] = None,
    rng_trace: Optional[TargetDiffRNGTrace] = None,
) -> CoupledSamplingResult:
    """Run the complete fixed-apo prelude and 20 PocketDiff-first blocks.

    ``state`` is interpreted as centered ``L_999``.  The function does not
    initialize ligand noise or recenter coordinates; those decisions belong to
    the caller and are therefore shared across baselines.
    """

    if not isinstance(adapter, TargetDiffAdapter):
        raise TypeError("adapter must be a TargetDiffAdapter")
    if not isinstance(pocket_solver, PocketStepSolver):
        raise TypeError("pocket_solver must be a PocketStepSolver")
    if not isinstance(state, TargetDiffState):
        raise TypeError("state must be a TargetDiffState")
    if not isinstance(residue_metadata, ResidueMetadata):
        raise TypeError("residue_metadata must be a ResidueMetadata")
    if adapter.num_timesteps != 1000:
        raise ValueError("full coupled sampling requires the official 1000-step TargetDiff schedule")
    if generator is not None and rng_trace is not None:
        raise ValueError("provide either generator or rng_trace, not both")

    before = _clone_state_tensors(state)
    prelude_timesteps = tuple(range(999, 199, -1))
    prelude_state, prelude_aux = adapter.run_steps(
        state,
        t_start=999,
        t_end_inclusive=200,
        generator=generator,
        rng_trace=rng_trace,
    )
    prelude_protein_unchanged = bool(torch.equal(prelude_state.protein_pos, before["protein_pos"]))

    current = prelude_state
    blocks = []
    for k in range(20):
        block = run_pocket_block(
            adapter,
            pocket_solver,
            current,
            residue_metadata,
            k,
            generator=generator,
            rng_trace=rng_trace,
        )
        blocks.append(block)
        current = block.state

    pocket_timesteps = tuple(block.event.pocket_targetdiff_t for block in blocks)
    block_targetdiff_timesteps = tuple(
        timestep for block in blocks for timestep in block.event.targetdiff_timesteps
    )
    targetdiff_timesteps = prelude_timesteps + block_targetdiff_timesteps
    event = CoupledSamplingEvent(
        prelude_timesteps=prelude_timesteps,
        pocket_timesteps=pocket_timesteps,
        block_targetdiff_timesteps=block_targetdiff_timesteps,
        targetdiff_timesteps=targetdiff_timesteps,
        targetdiff_call_count=len(prelude_aux) + sum(len(block.targetdiff_aux) for block in blocks),
        pocket_call_count=len(blocks),
        prelude_protein_unchanged=prelude_protein_unchanged,
        protein_features_unchanged=bool(torch.equal(current.protein_v, before["protein_v"])),
        apo_reference_unchanged=bool(torch.equal(current.apo_pos_ref, before["apo_pos_ref"])),
        batch_unchanged=bool(
            torch.equal(current.batch_protein, before["batch_protein"])
            and torch.equal(current.batch_ligand, before["batch_ligand"])
        ),
        center_offset_unchanged=bool(torch.equal(current.center_offset, before["center_offset"])),
        ligand_types_valid=bool(((current.ligand_v >= 0) & (current.ligand_v < 13)).all()),
    )
    return CoupledSamplingResult(
        state=current,
        prelude_aux=tuple(prelude_aux),
        blocks=tuple(blocks),
        event=event,
    )


__all__ = ["CoupledSamplingEvent", "CoupledSamplingResult", "run_coupled_sampling"]
