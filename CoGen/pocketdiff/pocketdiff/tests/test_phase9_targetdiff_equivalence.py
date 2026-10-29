"""Phase 9 tests for replayable TargetDiff reverse diffusion steps."""

import pytest
import torch
import torch.nn.functional as F

from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffRNGTrace,
    TargetDiffStepRandomness,
    initialize_targetdiff_state,
)


class _TraceFakeTargetDiff(torch.nn.Module):
    """Small deterministic model exposing the official posterior API."""

    def __init__(self, num_timesteps=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.num_classes = 13
        self.num_timesteps = num_timesteps
        self.posterior_logvar = torch.full((num_timesteps,), -2.0)

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
        del protein_pos, protein_v, batch_protein, init_ligand_v, batch_ligand, time_step
        logits = torch.zeros((init_ligand_pos.shape[0], 13), device=init_ligand_pos.device)
        logits[:, 0] = 2.0
        return {
            "pred_ligand_pos": init_ligand_pos + 0.25,
            "pred_ligand_v": logits,
        }

    def q_pos_posterior(self, x0, xt, t, batch):
        del t, batch
        return 0.5 * x0 + 0.5 * xt

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        del t, batch
        return torch.log_softmax(log_v0 + log_vt, dim=-1)


def _state():
    return initialize_targetdiff_state(
        protein_pos=torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        protein_v=torch.zeros(2, 27),
        batch_protein=torch.tensor([0, 0]),
        ligand_pos=torch.tensor([[2.0, 1.0, 0.0], [2.0, -1.0, 0.0]]),
        ligand_v=torch.tensor([1, 2]),
        batch_ligand=torch.tensor([0, 0]),
        apo_pos_ref=torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
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


def _trace_for(state, timesteps, seed=17):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return TargetDiffRNGTrace(
        {
            t: TargetDiffStepRandomness(
                position_noise=torch.randn(state.ligand_pos.shape, generator=generator),
                categorical_uniform=torch.rand(
                    (state.ligand_pos.shape[0], 13), generator=generator
                ),
            )
            for t in timesteps
        }
    )


def _reference_step(model, state, t, randomness):
    """Independent transcription of the official sample_diffusion formulas."""
    t_graph = torch.full((state.num_graphs,), t, dtype=torch.long)
    predictions = model(
        protein_pos=state.protein_pos,
        protein_v=state.protein_v,
        batch_protein=state.batch_protein,
        init_ligand_pos=state.ligand_pos,
        init_ligand_v=state.ligand_v,
        batch_ligand=state.batch_ligand,
        time_step=t_graph,
    )
    pred_x0 = predictions["pred_ligand_pos"]
    pred_v0_logits = predictions["pred_ligand_v"]
    pos_mean = model.q_pos_posterior(
        x0=pred_x0,
        xt=state.ligand_pos,
        t=t_graph,
        batch=state.batch_ligand,
    )
    pos_logvar = model.posterior_logvar[t_graph][state.batch_ligand].unsqueeze(-1)
    nonzero_mask = (1 - (t_graph == 0).float())[state.batch_ligand].unsqueeze(-1)
    next_pos = pos_mean + nonzero_mask * (0.5 * pos_logvar).exp() * randomness.position_noise

    log_v0 = F.log_softmax(pred_v0_logits, dim=-1)
    log_vt = torch.log(F.one_hot(state.ligand_v, 13).float().clamp(min=1e-30))
    log_model_prob = model.q_v_posterior(
        log_v0,
        log_vt,
        t_graph,
        state.batch_ligand,
    )
    gumbel = -torch.log(-torch.log(randomness.categorical_uniform + 1e-30) + 1e-30)
    next_v = (gumbel + log_model_prob).argmax(dim=-1)
    return state.replace(ligand_pos=next_pos, ligand_v=next_v), pred_x0, log_v0.exp(), log_model_prob.exp()


def test_trace_step_matches_independent_official_formula():
    model = _TraceFakeTargetDiff()
    adapter = TargetDiffAdapter(model)
    state = _state()
    trace = _trace_for(state, [3, 2, 1], seed=23)
    current = state
    for t in (3, 2, 1):
        expected, pred_x0, pred_v0_prob, posterior_prob = _reference_step(
            model, current, t, trace.for_step(t)
        )
        actual, aux = adapter.sample_step(current, t, rng_trace=trace.for_step(t))
        assert torch.equal(actual.ligand_v, expected.ligand_v)
        assert torch.equal(actual.ligand_pos, expected.ligand_pos)
        assert torch.equal(aux.pred_x0, pred_x0)
        assert torch.equal(aux.pred_v0_prob, pred_v0_prob)
        assert torch.equal(aux.posterior_v_prev_prob, posterior_prob)
        current = actual


def test_trace_and_generator_consume_position_then_categorical_randomness_identically():
    model = _TraceFakeTargetDiff()
    adapter = TargetDiffAdapter(model)
    state = _state()
    generator = torch.Generator(device="cpu").manual_seed(101)
    generated_state, generated_aux = adapter.run_steps(
        state, t_start=3, t_end_inclusive=1, generator=generator
    )

    trace_generator = torch.Generator(device="cpu").manual_seed(101)
    trace = {}
    for t in (3, 2, 1):
        # This order mirrors official sample_diffusion: position noise first,
        # categorical uniform second.
        trace[t] = TargetDiffStepRandomness(
            position_noise=torch.randn(state.ligand_pos.shape, generator=trace_generator),
            categorical_uniform=torch.rand((state.ligand_pos.shape[0], 13), generator=trace_generator),
        )
    traced_state, traced_aux = adapter.run_steps(
        state,
        t_start=3,
        t_end_inclusive=1,
        rng_trace=TargetDiffRNGTrace(trace),
    )
    assert torch.equal(generated_state.ligand_pos, traced_state.ligand_pos)
    assert torch.equal(generated_state.ligand_v, traced_state.ligand_v)
    for generated, traced in zip(generated_aux, traced_aux):
        assert torch.equal(generated.pred_x0, traced.pred_x0)
        assert torch.equal(generated.pred_v0_prob, traced.pred_v0_prob)
        assert torch.equal(generated.posterior_v_prev_prob, traced.posterior_v_prev_prob)


def test_trace_validates_missing_steps_shapes_and_exclusive_random_sources():
    model = _TraceFakeTargetDiff()
    adapter = TargetDiffAdapter(model)
    state = _state()
    valid = TargetDiffStepRandomness(
        position_noise=torch.zeros_like(state.ligand_pos),
        categorical_uniform=torch.full((state.ligand_pos.shape[0], 13), 0.5),
    )
    with pytest.raises(KeyError, match="no randomness for timestep 2"):
        adapter.run_steps(
            state,
            t_start=3,
            t_end_inclusive=1,
            rng_trace=TargetDiffRNGTrace({3: valid, 1: valid}),
        )
    with pytest.raises(ValueError, match="different atom counts"):
        TargetDiffStepRandomness(
            position_noise=torch.zeros(1, 3),
            categorical_uniform=torch.full((2, 13), 0.5),
        )
    with pytest.raises(ValueError, match="position_noise shape"):
        adapter.sample_step(
            state,
            3,
            rng_trace=TargetDiffStepRandomness(
                position_noise=torch.zeros(1, 3),
                categorical_uniform=torch.full((1, 13), 0.5),
            ),
        )
    with pytest.raises(ValueError, match="either generator or rng_trace"):
        adapter.sample_step(
            state,
            3,
            generator=torch.Generator(device="cpu"),
            rng_trace=valid,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        TargetDiffStepRandomness(
            position_noise=torch.zeros_like(state.ligand_pos),
            categorical_uniform=torch.ones((state.ligand_pos.shape[0], 13)),
        )


def test_full_fake_schedule_exposes_requested_snapshot_timesteps():
    model = _TraceFakeTargetDiff(num_timesteps=1000)
    adapter = TargetDiffAdapter(model)
    state = _state()
    zero_trace = TargetDiffRNGTrace(
        {
            t: TargetDiffStepRandomness(
                position_noise=torch.zeros_like(state.ligand_pos),
                categorical_uniform=torch.full((state.ligand_pos.shape[0], 13), 0.5),
            )
            for t in range(1000)
        }
    )
    snapshots = {}
    current = state
    for t in range(999, -1, -1):
        current, _ = adapter.sample_step(current, t, rng_trace=zero_trace.for_step(t))
        if t in {999, 200, 199, 189, 9, 0}:
            snapshots[t] = (current.ligand_pos.clone(), current.ligand_v.clone())
    assert list(sorted(snapshots, reverse=True)) == [999, 200, 199, 189, 9, 0]
    assert all(torch.isfinite(pos).all() for pos, _ in snapshots.values())
    assert all(((value >= 0) & (value < 13)).all() for _, value in snapshots.values())
