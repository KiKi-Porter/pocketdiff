import pytest
import torch
from pocketdiff.models import PocketDiffModel
from pocketdiff.tests.test_current_joint_model import inputs


def test_scalar_backend_is_default_and_legacy_forward_runs():
    state = inputs()
    model = PocketDiffModel(encoder_layers=1, knn=4, dropout=0., predict_chi=True, chi_input_mode='current')
    out = model(**state.model_kwargs())
    assert model.encoder_backend == 'scalar'
    assert model.input_contract == 'pocketdiff-current-joint-v1:encoder=scalar'
    assert out.remaining_chi.shape == (2, 5)


def test_targetdiff_backend_has_same_prediction_contract_without_coordinate_update():
    state = inputs()
    model = PocketDiffModel(encoder_backend='targetdiff', dropout=0., predict_chi=True, chi_input_mode='current').eval()
    before = state.protein_pos.clone()
    with torch.no_grad():
        out = model(**state.model_kwargs())
    assert out.remaining_translation_local.shape == (2, 3)
    assert out.remaining_rotvec_local.shape == (2, 3)
    assert out.remaining_chi.shape == (2, 5)
    assert torch.equal(state.protein_pos, before)
    assert torch.isfinite(out.remaining_translation_local).all()


def test_backend_strict_reload_and_mismatch_are_explicit():
    scalar = PocketDiffModel(encoder_layers=1, knn=4, predict_chi=True, chi_input_mode='current', dropout=0.)
    target = PocketDiffModel(encoder_backend='targetdiff', predict_chi=True, chi_input_mode='current', dropout=0.)
    scalar.load_state_dict(scalar.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        target.load_state_dict(scalar.state_dict(), strict=True)


def test_target_backend_rejects_scalar_only_options():
    with pytest.raises(ValueError, match='frozen'):
        PocketDiffModel(encoder_backend='targetdiff', hidden_dim=64)
    with pytest.raises(ValueError, match='scalar or targetdiff'):
        PocketDiffModel(encoder_backend='bad')
