import pytest
import torch

from pocketdiff.diffusion import DiffusionMotionAdapter
from pocketdiff.training.diffusion import build_scheduled_self_state


def test_scheduled_self_state_is_detached_and_time_aligned():
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    value = _synthetic_complex()
    model = DiffusionMotionAdapter(dropout=0.0)
    state = build_scheduled_self_state(model, value, 2, 8)
    assert state.model_input.diffusion_time.tolist() == [0.75]
    assert torch.equal(state.model_input.protein_pos, value.protein_pos_apo)
    assert not state.model_input.protein_pos.requires_grad
    assert torch.isfinite(state.target.protein_pos_holo).all()


def test_zero_output_self_state_is_stationary_at_every_exposure():
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    value = _synthetic_complex()
    model = DiffusionMotionAdapter(dropout=0.0)
    state = build_scheduled_self_state(model, value, 0, 4)
    exposed = build_scheduled_self_state(model, value, 4, 4)
    assert torch.equal(state.model_input.protein_pos, value.protein_pos_apo)
    assert torch.equal(exposed.model_input.protein_pos, value.protein_pos_apo)
    assert exposed.model_input.diffusion_time.tolist() == [0.0]


def test_exposure_arguments_are_validated():
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    value = _synthetic_complex()
    model = DiffusionMotionAdapter(dropout=0.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        build_scheduled_self_state(model, value, 5, 4)
