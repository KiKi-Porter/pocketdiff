import math

import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.geometry import ChiUpdateResult, apply_chi_updates
from pocketdiff.models import PocketDiffModel, ResidueChiHead
from pocketdiff.tests.test_model import _batch_inputs
from pocketdiff.tests.test_training import _batch
from pocketdiff.training import masked_periodic_chi_loss


def _chi_inputs():
    values = _batch_inputs()
    values["chi_apo"] = torch.tensor(
        [[math.pi - 0.1, 0.2, 0.0, 0.0, 0.0],
         [-math.pi + 0.1, -0.3, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    values["chi_mask"] = torch.tensor(
        [[True, True, False, False, False],
         [True, False, False, False, False]],
        dtype=torch.bool,
    )
    return values


def test_clean_batch_carries_chi_but_does_not_leak_holo_label():
    batch = _batch()
    assert batch.chi_apo.shape == (2, 5)
    assert batch.chi_holo.shape == (2, 5)
    assert batch.chi_mask.shape == (2, 5)
    kwargs = batch.model_kwargs()
    assert "chi_apo" in kwargs and "chi_mask" in kwargs
    assert "chi_holo" not in kwargs
    assert "target_chi" not in kwargs


def test_predict_chi_head_contract_and_zero_initialization():
    inputs = _chi_inputs()
    names = inputs.pop("protein_atom_name")
    model = PocketDiffModel(encoder_layers=1, knn=4, predict_chi=True)
    output = model(**inputs, protein_atom_name=names)
    assert output.remaining_chi.shape == (2, 5)
    assert torch.equal(output.remaining_chi[~inputs["chi_mask"]], torch.zeros(7))
    assert torch.equal(output.remaining_chi[inputs["chi_mask"]], torch.zeros(3))
    assert bool((output.remaining_chi.abs() <= math.pi).all())
    assert torch.isfinite(output.remaining_chi).all()


def test_chi_head_has_finite_nonzero_gradient_at_zero():
    head = ResidueChiHead()
    descriptor = torch.randn(2, 283, requires_grad=True)
    chi = torch.tensor([[math.pi, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, False, False, False], [True, False, False, False, False]])
    output = head(descriptor, chi, mask)
    target = torch.tensor([0.2, -0.1])
    loss = (output[:, 0] - target).square().sum()
    loss.backward()
    assert head.network[-1].weight.grad is not None
    assert torch.isfinite(head.network[-1].weight.grad).all()
    assert float(head.network[-1].weight.grad.abs().sum()) > 0.0


def test_periodic_chi_loss_uses_shortest_difference_and_masks_frames():
    prediction = PocketDiffPrediction(
        remaining_translation_local=torch.zeros(2, 3),
        remaining_rotvec_local=torch.zeros(2, 3),
        remaining_chi=torch.tensor([[0.2, 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], requires_grad=True),
        frame_valid=torch.tensor([True, False]),
        diagnostics={},
    )
    apo = torch.tensor([[math.pi - 0.1, 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
    holo = torch.tensor([[-math.pi + 0.1, 0., 0., 0., 0.], [1., 0., 0., 0., 0.]])
    mask = torch.tensor([[True, False, False, False, False], [True, False, False, False, False]])
    result = masked_periodic_chi_loss(prediction, apo, holo, mask)
    assert result.valid_chi_count == 1
    assert result.valid_residue_count == 1
    # The wrapped target is +0.2, so the prediction is exact.
    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    result.loss.backward()
    assert torch.isfinite(prediction.remaining_chi.grad).all()


def test_periodic_chi_loss_rejects_empty_supervision():
    prediction = PocketDiffPrediction(
        remaining_translation_local=torch.zeros(1, 3),
        remaining_rotvec_local=torch.zeros(1, 3),
        remaining_chi=torch.zeros(1, 5),
        frame_valid=torch.ones(1, dtype=torch.bool),
        diagnostics={},
    )
    with pytest.raises(ValueError, match="no valid chi"):
        masked_periodic_chi_loss(prediction, torch.zeros(1, 5), torch.zeros(1, 5),
                                 torch.zeros(1, 5, dtype=torch.bool))


def test_explicit_chi_update_is_finite_differentiable_and_does_not_mutate_input():
    positions = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [1., 0., 1.]],
        requires_grad=True,
    )
    axis_start = torch.full((1, 5), -1, dtype=torch.long)
    axis_end = torch.full((1, 5), -1, dtype=torch.long)
    axis_start[0, 0] = 0
    axis_end[0, 0] = 1
    downstream = torch.zeros(1, 5, 4, dtype=torch.bool)
    downstream[0, 0, 2:] = True
    delta = torch.tensor([[math.pi / 2, 0., 0., 0., 0.]], requires_grad=True)
    before = positions.detach().clone()
    result = apply_chi_updates(positions, axis_start, axis_end, downstream, delta)
    assert isinstance(result, ChiUpdateResult)
    assert torch.equal(positions.detach(), before)
    assert bool(result.valid[0, 0])
    torch.testing.assert_close(result.positions[2], torch.tensor([1., 0., 1.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.positions[3], torch.tensor([1., -1., 0.]), atol=1e-6, rtol=0)
    assert torch.isfinite(result.positions).all()
    result.positions.square().sum().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()


def test_explicit_chi_update_zero_angle_is_bitwise_stationary():
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    axis_start = torch.tensor([[0, -1, -1, -1, -1]])
    axis_end = torch.tensor([[1, -1, -1, -1, -1]])
    downstream = torch.zeros(1, 5, 3, dtype=torch.bool)
    downstream[0, 0, 2] = True
    result = apply_chi_updates(positions, axis_start, axis_end, downstream, torch.zeros(1, 5))
    assert torch.equal(result.positions, positions)
    assert torch.equal(result.applied_chi, torch.zeros(1, 5))
