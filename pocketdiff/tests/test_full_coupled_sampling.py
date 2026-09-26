"""Phase 11 tests for the complete 1000-step coupled scheduler."""

import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction, ResidueMetadata
from pocketdiff.sampling import PocketStepSolver, run_coupled_sampling
from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffRNGTrace,
    TargetDiffStepRandomness,
    initialize_targetdiff_state,
)


class _LongSpyTargetDiff(torch.nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.num_classes = 13
        self.num_timesteps = num_timesteps
        self.posterior_logvar = torch.full((num_timesteps,), -2.0)
        self.calls = []

    def forward(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        time_step,
    ):
        del protein_v, batch_protein, init_ligand_v, batch_ligand
        self.calls.append((int(time_step.item()), protein_pos.clone()))
        logits = torch.zeros((init_ligand_pos.shape[0], 13), device=init_ligand_pos.device)
        logits[:, 0] = 2.0
        return {
            "pred_ligand_pos": init_ligand_pos + 0.001,
            "pred_ligand_v": logits,
        }

    def q_pos_posterior(self, x0, xt, t, batch):
        del t, batch
        return 0.5 * x0 + 0.5 * xt

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        del t, batch
        return torch.log_softmax(log_v0 + log_vt, dim=-1)


class _FixedPocketModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.translation = torch.nn.Parameter(torch.tensor([[0.2, 0.0, 0.0]]))
        self.rotvec = torch.nn.Parameter(torch.zeros(1, 3))

    def forward(self, **kwargs):
        num_residues = kwargs["residue_type"].shape[0]
        return PocketDiffPrediction(
            remaining_translation_local=self.translation.expand(num_residues, -1),
            remaining_rotvec_local=self.rotvec.expand(num_residues, -1),
            remaining_chi=None,
            frame_valid=kwargs["frame_valid"],
            diagnostics={},
        )


def _state_and_metadata():
    protein_pos = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    state = initialize_targetdiff_state(
        protein_pos=protein_pos,
        protein_v=torch.zeros(3, 27),
        batch_protein=torch.zeros(3, dtype=torch.long),
        ligand_pos=torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
        ligand_v=torch.tensor([1], dtype=torch.long),
        batch_ligand=torch.zeros(1, dtype=torch.long),
        apo_pos_ref=protein_pos,
    )
    metadata = ResidueMetadata(
        protein_atom_name=[["N", "CA", "C"]],
        protein_residue_name=[["ALA", "ALA", "ALA"]],
        residue_type=torch.tensor([0], dtype=torch.long),
        atom_to_residue_global=torch.zeros(3, dtype=torch.long),
        batch_residue=torch.zeros(1, dtype=torch.long),
        frame_valid_reference=torch.tensor([True]),
        chi_mask=torch.zeros(1, 5, dtype=torch.bool),
        chain_id=[["A"]],
        residue_sequence_id=[["1"]],
    )
    return state, metadata


def _clone_state(state):
    return state.replace(
        protein_pos=state.protein_pos.clone(),
        protein_v=state.protein_v.clone(),
        batch_protein=state.batch_protein.clone(),
        ligand_pos=state.ligand_pos.clone(),
        ligand_v=state.ligand_v.clone(),
        batch_ligand=state.batch_ligand.clone(),
        apo_pos_ref=state.apo_pos_ref.clone(),
        center_offset=state.center_offset.clone(),
    )


def _trace(state, missing=None):
    return TargetDiffRNGTrace(
        {
            t: TargetDiffStepRandomness(
                position_noise=torch.zeros_like(state.ligand_pos),
                categorical_uniform=torch.full((state.ligand_pos.shape[0], 13), 0.5),
            )
            for t in range(1000)
            if t != missing
        }
    )


def test_full_scheduler_has_exact_schedule_counts_and_pocket_boundaries():
    state, metadata = _state_and_metadata()
    model = _LongSpyTargetDiff()
    adapter = TargetDiffAdapter(model)
    result = run_coupled_sampling(
        adapter,
        PocketStepSolver(_FixedPocketModel()),
        state,
        metadata,
        rng_trace=_trace(state),
    )
    expected_times = list(range(999, -1, -1))
    assert [call[0] for call in model.calls] == expected_times
    assert result.event.targetdiff_call_count == 1000
    assert result.event.pocket_call_count == 20
    assert result.event.prelude_timesteps == tuple(range(999, 199, -1))
    assert result.event.pocket_timesteps == tuple(199 - 10 * k for k in range(20))
    assert len(result.prelude_aux) == 800
    assert len(result.blocks) == 20
    assert all(len(block.targetdiff_aux) == 10 for block in result.blocks)
    assert result.event.prelude_protein_unchanged
    assert result.event.protein_features_unchanged
    assert result.event.apo_reference_unchanged
    assert result.event.batch_unchanged
    assert result.event.center_offset_unchanged
    assert result.event.ligand_types_valid
    assert torch.equal(model.calls[799][1], state.protein_pos)
    assert torch.equal(model.calls[800][1], result.blocks[0].pocket_output.protein_pos_next)
    assert torch.equal(model.calls[810][1], result.blocks[1].pocket_output.protein_pos_next)


def test_full_scheduler_is_replayable_and_does_not_mutate_input():
    state, metadata = _state_and_metadata()
    before = {name: getattr(state, name).clone() for name in (
        "protein_pos", "protein_v", "batch_protein", "ligand_pos", "ligand_v",
        "batch_ligand", "apo_pos_ref", "center_offset"
    )}
    first_model = _LongSpyTargetDiff()
    second_model = _LongSpyTargetDiff()
    trace = _trace(state)
    first = run_coupled_sampling(
        TargetDiffAdapter(first_model),
        PocketStepSolver(_FixedPocketModel()),
        _clone_state(state),
        metadata,
        rng_trace=trace,
    )
    second = run_coupled_sampling(
        TargetDiffAdapter(second_model),
        PocketStepSolver(_FixedPocketModel()),
        _clone_state(state),
        metadata,
        rng_trace=trace,
    )
    assert torch.equal(first.state.protein_pos, second.state.protein_pos)
    assert torch.equal(first.state.ligand_pos, second.state.ligand_pos)
    assert torch.equal(first.state.ligand_v, second.state.ligand_v)
    assert all(torch.equal(getattr(state, name), value) for name, value in before.items())


def test_full_scheduler_rejects_non_official_schedule_and_missing_trace():
    state, metadata = _state_and_metadata()
    with pytest.raises(ValueError, match="1000-step"):
        run_coupled_sampling(
            TargetDiffAdapter(_LongSpyTargetDiff(num_timesteps=200)),
            PocketStepSolver(_FixedPocketModel()),
            state,
            metadata,
            rng_trace=_trace(state),
        )
    with pytest.raises(KeyError, match="no randomness for timestep 998"):
        run_coupled_sampling(
            TargetDiffAdapter(_LongSpyTargetDiff()),
            PocketStepSolver(_FixedPocketModel()),
            state,
            metadata,
            rng_trace=_trace(state, missing=998),
        )
