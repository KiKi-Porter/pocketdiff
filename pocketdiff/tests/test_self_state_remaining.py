import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.models import PocketDiffModel
from pocketdiff.tests.test_training import _batch
from pocketdiff.training import build_self_state_batch, masked_self_state_motion_loss


def test_self_state_loss_is_physical_remaining_without_rate_scaling():
    prediction = PocketDiffPrediction(torch.tensor([[0.2, 0., 0.]]),
                                      torch.tensor([[0.1, 0., 0.]]), None,
                                      torch.tensor([True]), {})
    target_t = torch.tensor([[0.1, 0., 0.]])
    target_r = torch.tensor([[0.05, 0., 0.]])
    loss = masked_self_state_motion_loss(prediction, target_t, target_r, torch.tensor([True]))
    assert torch.allclose(loss.translation_loss, torch.tensor((.1**2)/3))
    assert torch.allclose(loss.rotation_loss, torch.tensor((.05**2)/3))
    assert torch.allclose(loss.loss, loss.translation_loss + loss.rotation_loss)


def test_remaining_mode_zero_head_has_finite_nonzero_self_state_gradients_at_k19():
    clean = _batch()
    trajectory = torch.stack([clean.protein_pos for _ in range(21)])
    state = build_self_state_batch(clean, trajectory, torch.tensor([19, 19]))
    model = PocketDiffModel(encoder_layers=1, knn=4, motion_parameterization='remaining')
    pred = model(**state.model_kwargs())
    loss = masked_self_state_motion_loss(pred, state.target_translation_local,
                                         state.target_rotvec_local, state.frame_valid)
    loss.loss.backward()
    grad = model.motion_head.network[-1].weight.grad
    assert torch.isfinite(loss.loss)
    assert torch.isfinite(grad).all()
    assert float(grad[:3].abs().sum()) > 0
    assert float(grad[3:].abs().sum()) > 0


def test_k0_self_state_loss_matches_clean_remaining_loss():
    from pocketdiff.training import masked_remaining_motion_loss
    clean = _batch()
    trajectory = torch.stack([clean.protein_pos for _ in range(21)])
    state = build_self_state_batch(clean, trajectory, torch.zeros(2, dtype=torch.long))
    model = PocketDiffModel(encoder_layers=1, knn=4)
    pred = model(**state.model_kwargs())
    a = masked_self_state_motion_loss(pred, state.target_translation_local,
                                      state.target_rotvec_local, state.frame_valid)
    b = masked_remaining_motion_loss(pred, clean.target_translation_local,
                                     clean.target_rotvec_local, clean.frame_valid)
    assert torch.equal(a.loss, b.loss)
    assert a.valid_residue_count == b.valid_residue_count


def test_remaining_output_does_not_impose_pi_r_bound():
    clean = _batch()
    # A finite local rotation below pi but above pi*r at k=19 is a valid
    # physical remaining target for a self-state that is behind schedule.
    target = torch.tensor([[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]])
    pred = PocketDiffPrediction(torch.zeros(2, 3), target*0, None,
                                torch.tensor([True, True]), {})
    loss = masked_self_state_motion_loss(pred, torch.zeros(2, 3), target,
                                         torch.tensor([True, True]))
    assert torch.isfinite(loss.loss)
    assert float(target.norm(dim=-1).max()) > torch.pi * .05
    state = build_self_state_batch(clean, clean.protein_pos.expand(21, -1, -1), torch.tensor([19, 19]))
    model = PocketDiffModel(encoder_layers=1, knn=4).eval()
    with torch.no_grad():
        model.motion_head.network[-1].bias[5] = torch.atanh(torch.tensor(.4/torch.pi))
    prediction = model(**state.model_kwargs())
    assert torch.allclose(prediction.remaining_rotvec_local, target, atol=1e-6)


def test_all_state_tensors_detached_and_old_next_prediction_is_not_a_target():
    from dataclasses import replace
    clean = _batch()
    clean = replace(clean, protein_pos_holo=clean.protein_pos_holo.clone().requires_grad_(),
                    ligand_pos=clean.ligand_pos.clone().requires_grad_())
    trajectory = clean.protein_pos.expand(21, -1, -1).clone().requires_grad_()
    batch = build_self_state_batch(clean, trajectory, torch.tensor([18, 19]))
    assert not hasattr(batch, 'protein_pos_next_target')
    model = PocketDiffModel(encoder_layers=1, knn=4)
    pred = model(**batch.model_kwargs())
    loss = masked_self_state_motion_loss(pred, batch.target_translation_local,
                                         batch.target_rotvec_local, batch.frame_valid)
    loss.loss.backward()
    assert trajectory.grad is None
    assert clean.protein_pos_holo.grad is None
    assert clean.ligand_pos.grad is None
    assert all(not v.requires_grad for v in batch.__dict__.values() if isinstance(v, torch.Tensor))
    assert model.motion_head.network[-1].weight.grad.abs().sum() > 0


def test_self_state_cannot_recycle_non_endpoint_input_or_drop_valid_frames():
    from dataclasses import replace
    clean = _batch()
    trajectory = clean.protein_pos.expand(21, -1, -1).clone()
    with pytest.raises(ValueError, match='apo endpoints'):
        build_self_state_batch(replace(clean, protein_pos=clean.protein_pos+1), trajectory, torch.tensor([0, 0]))
    trajectory[19, :4] = 0
    with pytest.raises(ValueError, match='invalidated'):
        build_self_state_batch(clean, trajectory, torch.tensor([19, 19]))


def test_unknown_parameterization_still_rejected_and_legacy_default_is_remaining():
    assert PocketDiffModel().motion_parameterization == 'remaining'
    with pytest.raises(ValueError, match='motion_parameterization'):
        PocketDiffModel(motion_parameterization='self_state_magic')
