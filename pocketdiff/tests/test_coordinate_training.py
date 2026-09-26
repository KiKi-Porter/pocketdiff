from dataclasses import replace

import pytest
import torch

from pocketdiff.models import PocketDiffModel
from pocketdiff.tests.test_current_joint_model import inputs
from pocketdiff.training import NextXYZBatch, next_xyz_objective, train_next_xyz_step


def make_batch():
    state = inputs()  # two independent graphs at k=0 and k=19
    target = state.protein_pos + torch.tensor([.08, -.03, .05])
    target = target.clone()
    target[[5, 11], 2] += .15
    return NextXYZBatch(state, target, torch.ones(2, dtype=torch.bool))


def make_model(backend):
    return PocketDiffModel(encoder_backend=backend, encoder_layers=1, knn=4,
                           predict_chi=True, chi_input_mode='current', dropout=0.)


@pytest.mark.parametrize('backend', ['scalar', 'targetdiff'])
def test_coordinate_step_trains_heads_then_encoder_without_changing_inputs(backend):
    torch.manual_seed(35)
    batch = make_batch()
    before = {key: val.clone() for key, val in vars(batch.inputs).items()
              if isinstance(val, torch.Tensor)}
    network = make_model(backend).eval()
    encoder_before = {key: val.clone() for key, val in network.encoder.state_dict().items()}
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    first = train_next_xyz_step(network, batch, optimizer)
    assert first.valid_atom_count == 12 and first.valid_graph_count == 2
    assert all(first.gradient_norms[key] > 0 for key in ('translation', 'rotation', 'chi'))
    assert first.gradient_norms['encoder'] == 0.
    second = train_next_xyz_step(network, batch, optimizer)
    assert second.gradient_norms['encoder'] > 0
    assert second.loss_before_update < first.loss_before_update
    assert any(not torch.equal(val, encoder_before[key])
               for key, val in network.encoder.state_dict().items())
    assert all(torch.equal(getattr(batch.inputs, key), val) for key, val in before.items())
    assert not network.training


def test_supervision_changes_loss_but_never_prediction_or_label_gradients():
    batch = make_batch()
    network = make_model('scalar').eval()
    with torch.no_grad():
        for head in (network.motion_head, network.current_chi_head):
            head.network[-1].weight.normal_(0., .003)
            head.network[-1].bias.fill_(.01)
    target = batch.target_pos_next.clone().requires_grad_()
    first = next_xyz_objective(network, replace(batch, target_pos_next=target))
    second = next_xyz_objective(network, replace(
        batch, target_pos_next=target + 1., supervision_frame_valid=torch.tensor([True, False])))
    assert torch.equal(first.step_output.protein_pos_next, second.step_output.protein_pos_next)
    assert not torch.equal(first.step_output.protein_pos_next, batch.inputs.protein_pos)
    assert second.coordinate.valid_graph_count == 1
    assert first.coordinate.loss != second.coordinate.loss
    first.coordinate.loss.backward()
    assert target.grad is None


@pytest.mark.parametrize('bad', ['empty_mask', 'nan_target', 'foreign_optimizer'])
def test_invalid_supervision_or_optimizer_rejected_before_parameters_change(bad):
    batch = make_batch()
    network = make_model('scalar').eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    if bad == 'empty_mask':
        batch = replace(batch, supervision_frame_valid=torch.zeros(2, dtype=torch.bool))
    elif bad == 'nan_target':
        batch = replace(batch, target_pos_next=torch.full_like(batch.target_pos_next, float('nan')))
    else:
        optimizer = torch.optim.Adam(make_model('scalar').parameters())
    before = {key: val.clone() for key, val in network.state_dict().items()}
    with pytest.raises(ValueError):
        train_next_xyz_step(network, batch, optimizer)
    assert all(torch.equal(val, before[key]) for key, val in network.state_dict().items())
    assert not network.training
    assert len(optimizer.state) == 0


def test_nonfinite_gradient_does_not_advance_optimizer_and_restores_mode():
    network = make_model('scalar').eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    before = {key: val.clone() for key, val in network.state_dict().items()}
    hook = network.motion_head.network[-1].weight.register_hook(
        lambda gradient: torch.full_like(gradient, float('nan')))
    try:
        with pytest.raises(FloatingPointError, match='gradient'):
            train_next_xyz_step(network, make_batch(), optimizer)
    finally:
        hook.remove()
    assert all(torch.equal(val, before[key]) for key, val in network.state_dict().items())
    assert not network.training
    assert len(optimizer.state) == 0
