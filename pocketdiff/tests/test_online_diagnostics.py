import torch

from pocketdiff.evaluation import (
    summarize_online_self_state_scales,
    summarize_online_trajectory,
)
from pocketdiff.models import PocketDiffModel
from pocketdiff.tests.test_training import _batch
from pocketdiff.training import refresh_online_trajectory


def test_online_diagnostics_are_finite_and_repeatable():
    clean = _batch()
    model = PocketDiffModel(encoder_layers=1, knn=4).train()
    trajectory = refresh_online_trajectory(model, clean)
    before_input = {name: value.clone() for name, value in clean.__dict__.items()
                    if isinstance(value, torch.Tensor)}
    before_state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    first_trajectory = summarize_online_trajectory(clean, trajectory)
    first_scale = summarize_online_self_state_scales(model, clean, trajectory)
    second_trajectory = summarize_online_trajectory(clean, trajectory)
    second_scale = summarize_online_self_state_scales(model, clean, trajectory)
    assert first_trajectory == second_trajectory
    assert first_scale == second_scale
    assert len(first_trajectory['per_step']) == 21
    assert len(first_scale['per_k']) == 20
    assert all(torch.isfinite(value).all()
               for value in model.state_dict().values())
    for name, value in before_state.items():
        assert torch.equal(model.state_dict()[name], value)
    for name, value in before_input.items():
        assert torch.equal(getattr(clean, name), value)
    assert model.training
    assert torch.equal(rng, torch.get_rng_state())
