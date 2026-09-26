"""Check diagnostic evidence against independent geometry and state guards."""
import pytest
import torch

from pocketdiff.tests.test_joint_trajectory import example
from pocketdiff.tests.test_current_joint_model import model
from pocketdiff.training import build_joint_reference
from pocketdiff.scripts.phase37_fixed_checkpoint_diagnosis import diagnose_batch


def test_diagnosis_preserves_mode_rng_parameters_and_preexisting_gradients():
    state, holo = example()
    ref = build_joint_reference(state, holo)
    network = model(nonzero=True).train()
    for p in network.parameters():
        p.grad = torch.ones_like(p)
    weights = {k: v.clone() for k, v in network.state_dict().items()}
    rng = torch.get_rng_state().clone()
    row, gradients = diagnose_batch(network, ref.batch_at(torch.tensor([0, 0])), ref.positions[-1])
    assert network.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, weights[k]) for k, v in network.state_dict().items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in network.parameters())
    assert torch.isfinite(gradients).all() and gradients.norm() > 0
    assert row['scaled_output_loss'] == pytest.approx(row['loss'], abs=1e-9)


def test_oracle_remaining_shrinks_while_per_step_target_is_constant():
    state, holo = example()
    ref = build_joint_reference(state, holo)
    network = model()
    early, _ = diagnose_batch(network, ref.batch_at(torch.tensor([0, 0])), ref.positions[-1])
    late, _ = diagnose_batch(network, ref.batch_at(torch.tensor([19, 19])), ref.positions[-1])
    for head in ('translation', 'rotation', 'chi'):
        assert early['motion'][head]['target_remaining_rms'] == pytest.approx(
            20 * late['motion'][head]['target_remaining_rms'], rel=5e-4)
        assert early['motion'][head]['target_step_rms'] == pytest.approx(
            late['motion'][head]['target_step_rms'], rel=5e-4)
    assert late['loss'] == late['zero_loss'] == late['scaled_output_loss']
    assert late['displacement_cosine'] is None  # zero vector is not aligned


def test_diagnosis_restores_mode_after_failure(monkeypatch):
    state, holo = example()
    ref = build_joint_reference(state, holo)
    network = model().train()
    def fail(**kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(network, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected failure'):
        diagnose_batch(network, ref.batch_at(torch.tensor([0, 0])), ref.positions[-1])
    assert network.training
