from dataclasses import replace

import pytest
import torch

from pocketdiff.models import PocketDiffModel
from pocketdiff.tests.test_training import _batch
from pocketdiff.training import (OnlineTrajectory, build_latest_self_state,
                                  refresh_online_trajectory)


def test_zero_model_refresh_is_deterministic_restores_mode_and_rng():
    clean = _batch()
    model = PocketDiffModel(encoder_layers=1, knn=4).train()
    before = {name: value.clone() for name, value in clean.__dict__.items()
              if isinstance(value, torch.Tensor)}
    rng = torch.get_rng_state().clone()
    first = refresh_online_trajectory(model, clean)
    second = refresh_online_trajectory(model, clean)
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.positions, clean.apo_pos_ref.expand(21, -1, -1))
    assert model.training
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in before.items():
        assert torch.equal(getattr(clean, name), value)


def test_refresh_uses_current_model_and_latest_batch_selects_mixed_k():
    clean = _batch()
    model = PocketDiffModel(encoder_layers=1, knn=4).eval()
    with torch.no_grad():
        model.motion_head.network[-1].bias[0] = .02
    trajectory = refresh_online_trajectory(model, clean)
    assert not torch.equal(trajectory.positions[-1], clean.apo_pos_ref)
    state = build_latest_self_state(clean, trajectory, torch.tensor([0, 19]))
    assert torch.equal(state.protein_pos[:4], trajectory.positions[0, :4])
    assert torch.equal(state.protein_pos[4:], trajectory.positions[19, 4:])
    assert not state.protein_pos.requires_grad
    assert not state.target_translation_local.requires_grad


def test_holo_and_supervision_changes_do_not_affect_refresh():
    clean = _batch()
    model = PocketDiffModel(encoder_layers=1, knn=4).eval()
    first = refresh_online_trajectory(model, clean)
    changed = replace(clean, protein_pos_holo=clean.protein_pos_holo + 100,
                      target_translation_local=clean.target_translation_local + 100,
                      target_rotvec_local=clean.target_rotvec_local + 100)
    second = refresh_online_trajectory(model, changed)
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.frame_valid, second.frame_valid)


def test_online_trajectory_rejects_invalid_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match='positions'):
        OnlineTrajectory(torch.zeros(20, 2, 3), torch.zeros(21, 2, dtype=torch.bool),
                         torch.zeros(20, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match='finite'):
        OnlineTrajectory(torch.full((21, 2, 3), float('nan')),
                         torch.ones(21, 1, dtype=torch.bool), torch.ones(20, 1, dtype=torch.bool))


def test_latest_state_rejects_wrong_trajectory_atom_count():
    clean = _batch()
    trajectory = refresh_online_trajectory(PocketDiffModel(encoder_layers=1, knn=4), clean)
    wrong = OnlineTrajectory(trajectory.positions[:, :-1], trajectory.frame_valid,
                             trajectory.update_valid)
    with pytest.raises(ValueError, match='atom count'):
        build_latest_self_state(clean, wrong, torch.tensor([0, 0]))
