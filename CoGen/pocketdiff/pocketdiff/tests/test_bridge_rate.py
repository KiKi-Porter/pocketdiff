from dataclasses import replace

import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.geometry.so3 import so3_exp
from pocketdiff.models import PocketDiffModel
from pocketdiff.tests.test_model import _batch_inputs
from pocketdiff.tests.test_training import _batch
from pocketdiff.training import MultiKCleanTrainer, build_bridge_batch, masked_bridge_rate_loss


def test_mixed_times_decode_to_actual_remaining_without_changing_weights():
    torch.manual_seed(17)
    old = PocketDiffModel(encoder_layers=1, knn=4).eval()
    torch.manual_seed(17)
    rate = PocketDiffModel(encoder_layers=1, knn=4, motion_parameterization='bridge_rate').eval()
    assert old.state_dict().keys() == rate.state_dict().keys()
    assert all(torch.equal(v, rate.state_dict()[n]) for n, v in old.state_dict().items())
    inputs = _batch_inputs()
    inputs['pocket_k'] = torch.tensor([0, 19])
    zero = rate(**inputs)
    assert torch.count_nonzero(zero.remaining_translation_local) == 0
    assert torch.count_nonzero(zero.remaining_rotvec_local) == 0
    with torch.no_grad():
        old.motion_head.network[-1].weight.normal_(std=.02)
        old.motion_head.network[-1].bias.normal_(std=.01)
    rate.load_state_dict(old.state_dict(), strict=True)
    raw, scaled = old(**inputs), rate(**inputs)
    fraction = torch.tensor([[1.], [.05]])
    for name in ('remaining_translation_local', 'remaining_rotvec_local'):
        assert torch.equal(getattr(scaled, name), getattr(raw, name) * fraction)
    rotation = so3_exp(torch.tensor([.2, -.1, .3]))
    moved = dict(inputs)
    for name in ('protein_pos', 'apo_pos_ref', 'ligand_pos'):
        moved[name] = inputs[name] @ rotation.T + torch.tensor([2., -1., 3.])
    transformed = rate(**moved)
    for name in ('remaining_translation_local', 'remaining_rotvec_local'):
        assert torch.allclose(getattr(scaled, name), getattr(transformed, name), atol=2e-5)
    inputs['frame_valid'] = torch.tensor([True, False])
    masked = rate(**inputs)
    assert torch.count_nonzero(masked.remaining_translation_local[1]) == 0
    assert torch.count_nonzero(masked.remaining_rotvec_local[1]) == 0


def test_equal_rate_error_has_equal_loss_and_head_gradient_at_early_and_late_times():
    results = []
    for k in (0, 19):
        rate = torch.tensor([[.2, -.1, .3]], requires_grad=True)
        r = (20-k)/20
        valid = torch.tensor([True])
        prediction = PocketDiffPrediction(rate*r, rate*r, None, valid, {})
        loss = masked_bridge_rate_loss(prediction, torch.ones(1, 3)*r,
                                       torch.ones(1, 3)*r, valid, torch.tensor([k]), torch.tensor([0]))
        loss.loss.backward()
        results.append((loss.loss.detach(), rate.grad))
    assert torch.allclose(results[0][0], results[1][0])
    assert torch.allclose(results[0][1], results[1][1])


def test_zero_head_learns_both_motion_channels_at_k19():
    batch = build_bridge_batch(_batch(), torch.tensor([19, 19]))
    model = PocketDiffModel(encoder_layers=1, knn=4, motion_parameterization='bridge_rate')
    pred = model(**batch.model_kwargs())
    loss = masked_bridge_rate_loss(pred, batch.target_translation_local, batch.target_rotvec_local,
                                   batch.frame_valid, batch.pocket_k, batch.batch_residue)
    loss.loss.backward()
    grad = model.motion_head.network[-1].weight.grad
    assert torch.isfinite(grad).all()
    assert grad[:3].abs().sum() > 0
    assert grad[3:].abs().sum() > 0


@pytest.mark.parametrize('bad_k', [-1, 20])
def test_invalid_time_rejected_by_model_and_loss(bad_k):
    inputs = _batch_inputs()
    model = PocketDiffModel(motion_parameterization='bridge_rate')
    pred = model(**inputs)
    inputs['pocket_k'] = torch.tensor([bad_k, 0])
    with pytest.raises(ValueError, match='pocket_k'):
        model(**inputs)
    with pytest.raises(ValueError, match='pocket_k'):
        masked_bridge_rate_loss(pred, torch.zeros(2, 3), torch.zeros(2, 3), pred.frame_valid,
                                inputs['pocket_k'], inputs['batch_residue'])


def test_malformed_targets_and_unknown_mode_rejected():
    with pytest.raises(ValueError, match='motion_parameterization'):
        PocketDiffModel(motion_parameterization='unknown')
    inputs = _batch_inputs()
    pred = PocketDiffModel()(**inputs)
    with pytest.raises(ValueError, match='shapes differ'):
        masked_bridge_rate_loss(pred, torch.zeros(1, 3), torch.zeros(2, 3), pred.frame_valid,
                                inputs['pocket_k'], inputs['batch_residue'])


@pytest.mark.parametrize('mode', ['remaining', 'bridge_rate'])
def test_checkpoint_replays_rate_training_and_legacy_config(tmp_path, mode):
    batch = _batch()
    trainer = MultiKCleanTrainer(batch, model_config=dict(encoder_layers=1, knn=4,
                                                         motion_parameterization=mode))
    for _ in range(3):
        trainer.step()
    path = tmp_path/'model.pt'
    trainer.save_checkpoint(path)
    payload = torch.load(path)
    assert payload['model_config']['motion_parameterization'] == mode
    if mode == 'remaining':
        del payload['model_config']['motion_parameterization']  # Phase17 format
        torch.save(payload, path)
    expected = [trainer.step() for _ in range(3)]
    restored = MultiKCleanTrainer.from_checkpoint(path, batch)
    assert restored.model.motion_parameterization == mode
    assert [restored.step() for _ in range(3)] == expected
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
