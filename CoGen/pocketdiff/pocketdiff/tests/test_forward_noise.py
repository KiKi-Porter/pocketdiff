from dataclasses import fields

import pytest
import torch

from pocketdiff.targetdiff import (
    TargetDiffAdapter, TargetDiffStepRandomness,
    forward_noise_reference, initialize_targetdiff_state,
)


class ScheduleModel(torch.nn.Module):
    num_classes = 13
    num_timesteps = 3

    def __init__(self):
        super().__init__()
        self.register_buffer("alphas_cumprod", torch.tensor([1.0, 0.25, 0.0]))

    def q_v_pred(self, log_v0, t, batch):
        a = self.alphas_cumprod[t][batch, None]
        return (a * log_v0.exp() + (1 - a) / 13).clamp_min(1e-30).log()

    def forward(self, **kwargs):
        raise AssertionError("forward noise must not run the neural network")


@pytest.fixture
def inputs():
    state = initialize_targetdiff_state(
        protein_pos=torch.tensor([[2., 0., 0.], [8., 0., 0.]]),
        protein_v=torch.zeros(2, 27), batch_protein=torch.tensor([0, 1]),
        ligand_pos=torch.tensor([[3., 1., 0.], [9., 2., 1.], [4., 0., 2.]]),
        ligand_v=torch.tensor([0, 12, 5]), batch_ligand=torch.tensor([0, 1, 0]),
    )
    return TargetDiffAdapter(ScheduleModel()), state


def test_mixed_graph_times_and_clean_noise_limits(inputs):
    adapter, state = inputs
    noise = torch.arange(9).float().reshape(3, 3)
    uniform = torch.full((3, 13), 0.5)
    uniform[:, 7] = 0.99
    trace = TargetDiffStepRandomness(noise, uniform)
    out = forward_noise_reference(adapter, state, torch.tensor([0, 2]), randomness=trace)
    assert torch.equal(out.ligand_pos[[0, 2]], state.ligand_pos[[0, 2]])
    assert torch.equal(out.ligand_v[[0, 2]], state.ligand_v[[0, 2]])
    assert torch.equal(out.ligand_pos[1], noise[1])
    assert out.ligand_v[1] == 7
    # Intermediate schedule verifies shrinkage around fixed apo center.
    middle = forward_noise_reference(adapter, state, torch.tensor([1, 1]), randomness=trace)
    torch.testing.assert_close(middle.ligand_pos, 0.5 * state.ligand_pos + (0.75 ** 0.5) * noise)


def test_generator_replay_preserves_inputs_and_global_rng(inputs):
    adapter, state = inputs
    before = {f.name: getattr(state, f.name).clone() for f in fields(state)}
    state.ligand_pos.requires_grad_(True)
    global_rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(31)
    replay = torch.Generator().manual_seed(31)
    noise = torch.randn((3, 3), generator=replay)
    uniform = torch.rand((3, 13), generator=replay)
    out = forward_noise_reference(adapter, state, torch.tensor([1, 2]), generator=generator)
    traced = forward_noise_reference(adapter, state, torch.tensor([1, 2]),
                                     randomness=TargetDiffStepRandomness(noise, uniform))
    assert torch.equal(out.ligand_pos, traced.ligand_pos)
    assert torch.equal(out.ligand_v, traced.ligand_v)
    assert torch.equal(generator.get_state(), replay.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert not out.ligand_pos.requires_grad
    for name, value in before.items():
        assert torch.equal(getattr(state, name), value)
        if name not in ('ligand_pos', 'ligand_v'):
            assert torch.equal(getattr(out, name), value)


@pytest.mark.parametrize('times', [torch.tensor([-1, 0]), torch.tensor([0, 3]),
                                  torch.tensor([0]), torch.tensor([0., 1.])])
def test_invalid_graph_times_rejected(inputs, times):
    adapter, state = inputs
    with pytest.raises(ValueError, match='t_graph'):
        forward_noise_reference(adapter, state, times)


def test_invalid_random_source_rejected(inputs):
    adapter, state = inputs
    trace = TargetDiffStepRandomness(torch.zeros(2, 3), torch.full((2, 13), 0.5))
    with pytest.raises(ValueError, match='atom count'):
        forward_noise_reference(adapter, state, torch.tensor([0, 1]), randomness=trace)
    with pytest.raises(ValueError, match='either'):
        forward_noise_reference(adapter, state, torch.tensor([0, 1]), randomness=trace,
                                generator=torch.Generator())
