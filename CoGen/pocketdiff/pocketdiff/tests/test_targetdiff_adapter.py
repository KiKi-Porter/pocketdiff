import torch
import pytest

from pocketdiff.targetdiff import TargetDiffAdapter, TargetDiffStepAux, initialize_targetdiff_state, restore_center


class _FakeTargetDiff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.num_classes = 13
        self.num_timesteps = 4
        self.posterior_logvar = torch.tensor([-2.0, -2.0, -2.0, -2.0])

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand, time_step):
        logits = torch.zeros((init_ligand_pos.shape[0], 13), device=init_ligand_pos.device)
        logits[:, 0] = 2.0
        return {"pred_ligand_pos": init_ligand_pos + 0.25, "pred_ligand_v": logits}

    def q_pos_posterior(self, x0, xt, t, batch):
        return 0.5 * x0 + 0.5 * xt

    def q_v_pred(self, log_v0, t, batch):
        return log_v0

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        return log_vt_1

    def q_v_posterior(self, log_v0, log_vt, t, batch):
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


def test_initialize_state_centers_all_coordinates_once():
    state = _state()
    assert torch.allclose(state.protein_pos.mean(dim=0), torch.zeros(3))
    assert torch.allclose(state.apo_pos_ref, state.protein_pos)
    assert torch.allclose(state.center_offset, torch.tensor([[2.0, 0.0, 0.0]]))
    assert torch.equal(restore_center(state.protein_pos, state.batch_protein, state.center_offset), torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))


def test_sample_step_is_finite_and_does_not_mutate_protein_or_batch():
    adapter = TargetDiffAdapter(_FakeTargetDiff())
    state = _state()
    before = {name: getattr(state, name).clone() for name in (
        "protein_pos", "protein_v", "batch_protein", "batch_ligand", "apo_pos_ref", "center_offset"
    )}
    generator = torch.Generator(device="cpu").manual_seed(11)
    next_state, aux = adapter.sample_step(state, 3, generator=generator)
    assert isinstance(aux, TargetDiffStepAux)
    assert next_state.ligand_pos.shape == state.ligand_pos.shape
    assert next_state.ligand_v.shape == state.ligand_v.shape
    assert torch.isfinite(next_state.ligand_pos).all()
    assert torch.isfinite(aux.pred_x0).all()
    assert torch.equal(next_state.protein_pos, before["protein_pos"])
    assert torch.equal(next_state.protein_v, before["protein_v"])
    assert torch.equal(next_state.batch_protein, before["batch_protein"])
    assert torch.equal(next_state.batch_ligand, before["batch_ligand"])
    assert torch.equal(next_state.apo_pos_ref, before["apo_pos_ref"])
    assert torch.equal(next_state.center_offset, before["center_offset"])
    assert torch.equal(state.ligand_v, torch.tensor([1, 2]))


def test_t0_has_no_position_noise_but_still_returns_type_posterior():
    adapter = TargetDiffAdapter(_FakeTargetDiff())
    state = _state()
    generator = torch.Generator(device="cpu").manual_seed(3)
    next_state, aux = adapter.sample_step(state, 0, generator=generator)
    expected = 0.5 * (state.ligand_pos + 0.25) + 0.5 * state.ligand_pos
    assert torch.allclose(next_state.ligand_pos, expected)
    assert aux.posterior_v_prev_prob.shape == (2, 13)
    assert torch.allclose(aux.posterior_v_prev_prob.sum(dim=-1), torch.ones(2))


def test_sample_step_rejects_invalid_time():
    adapter = TargetDiffAdapter(_FakeTargetDiff())
    with pytest.raises(ValueError, match="integer"):
        adapter.sample_step(_state(), 1.5)
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        adapter.sample_step(_state(), 4)


def test_run_steps_descends_and_returns_auxiliaries():
    adapter = TargetDiffAdapter(_FakeTargetDiff())
    state, auxiliaries = adapter.run_steps(_state(), t_start=3, t_end_inclusive=1)
    assert len(auxiliaries) == 3
    assert state.ligand_pos.shape == (2, 3)
