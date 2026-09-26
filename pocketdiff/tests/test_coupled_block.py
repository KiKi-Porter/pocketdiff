"""Phase 10 tests for one PocketDiff -> TargetDiff coupling block."""

import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction, ResidueMetadata
from pocketdiff.sampling import PocketStepSolver, run_pocket_block
from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffRNGTrace,
    TargetDiffStepRandomness,
    initialize_targetdiff_state,
)


class _SpyTargetDiff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.num_classes = 13
        self.num_timesteps = 200
        self.posterior_logvar = torch.full((200,), -2.0)
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
        self.calls.append((time_step.clone(), protein_pos.clone()))
        logits = torch.zeros((init_ligand_pos.shape[0], 13), device=init_ligand_pos.device)
        logits[:, 0] = 2.0
        return {
            "pred_ligand_pos": init_ligand_pos + 0.05,
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
        self.translation = torch.nn.Parameter(torch.tensor([[0.4, 0.0, 0.0]]))
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


def _trace(state, start_t):
    return TargetDiffRNGTrace(
        {
            t: TargetDiffStepRandomness(
                position_noise=torch.zeros_like(state.ligand_pos),
                categorical_uniform=torch.full((state.ligand_pos.shape[0], 13), 0.5),
            )
            for t in range(start_t, start_t - 10, -1)
        }
    )


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


@pytest.mark.parametrize("k", [0, 19])
def test_block_schedule_and_order(k):
    state, metadata = _state_and_metadata()
    target_model = _SpyTargetDiff()
    adapter = TargetDiffAdapter(target_model)
    pocket_model = _FixedPocketModel()
    solver = PocketStepSolver(pocket_model)
    trace = _trace(state, 199 - 10 * k)
    result = run_pocket_block(adapter, solver, state, metadata, k, rng_trace=trace)
    expected_times = tuple(range(199 - 10 * k, 189 - 10 * k, -1))
    assert result.event.targetdiff_timesteps == expected_times
    assert len(result.targetdiff_aux) == 10
    assert result.event.pocket_targetdiff_t == 199 - 10 * k
    assert result.event.ligand_unchanged_during_pocket
    assert result.event.protein_features_unchanged
    assert result.event.apo_reference_unchanged
    assert result.event.batch_unchanged
    assert result.event.center_offset_unchanged
    assert [int(call[0].item()) for call in target_model.calls] == list(expected_times)
    assert torch.equal(target_model.calls[0][1], result.pocket_output.protein_pos_next)


def test_block_equals_manual_pocket_then_ten_targetdiff_steps():
    state, metadata = _state_and_metadata()
    target_model_block = _SpyTargetDiff()
    target_model_manual = _SpyTargetDiff()
    adapter_block = TargetDiffAdapter(target_model_block)
    adapter_manual = TargetDiffAdapter(target_model_manual)
    solver_block = PocketStepSolver(_FixedPocketModel())
    solver_manual = PocketStepSolver(_FixedPocketModel())
    trace = _trace(state, 199)

    result = run_pocket_block(
        adapter_block,
        solver_block,
        _clone_state(state),
        metadata,
        0,
        rng_trace=trace,
    )
    pocket_output = solver_manual.step(_clone_state(state), metadata, 0)
    manual_state, manual_aux = adapter_manual.run_steps(
        state.replace(protein_pos=pocket_output.protein_pos_next),
        t_start=199,
        t_end_inclusive=190,
        rng_trace=trace,
    )
    assert torch.equal(result.state.protein_pos, manual_state.protein_pos)
    assert torch.equal(result.state.ligand_pos, manual_state.ligand_pos)
    assert torch.equal(result.state.ligand_v, manual_state.ligand_v)
    assert len(manual_aux) == len(result.targetdiff_aux) == 10
    for actual, expected in zip(result.targetdiff_aux, manual_aux):
        assert torch.equal(actual.pred_x0, expected.pred_x0)
        assert torch.equal(actual.pred_v0_prob, expected.pred_v0_prob)
        assert torch.equal(actual.posterior_v_prev_prob, expected.posterior_v_prev_prob)


def test_block_rejects_asynchronous_k_and_two_random_sources():
    state, metadata = _state_and_metadata()
    adapter = TargetDiffAdapter(_SpyTargetDiff())
    solver = PocketStepSolver(_FixedPocketModel())
    trace = _trace(state, 199)
    with pytest.raises(ValueError, match="scalar LongTensor"):
        run_pocket_block(
            adapter,
            solver,
            state,
            metadata,
            torch.tensor([0], dtype=torch.long),
            rng_trace=trace,
        )
    with pytest.raises(ValueError, match="either generator or rng_trace"):
        run_pocket_block(
            adapter,
            solver,
            state,
            metadata,
            0,
            generator=torch.Generator(device="cpu"),
            rng_trace=trace,
        )
