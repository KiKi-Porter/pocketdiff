import torch
import pytest

from pocketdiff.tests.test_training import _batch
from pocketdiff.training import OnlineSelfStateTrainer


def _trainer(batch, **kwargs):
    return OnlineSelfStateTrainer(
        batch,
        model_config={'encoder_layers': 1, 'knn': 4},
        seed=7,
        **kwargs,
    )


def test_online_trainer_refreshes_after_fixed_interval():
    batch = _batch()
    trainer = _trainer(batch, refresh_interval=2)
    first = trainer.step()
    assert first['refreshed'] is False
    assert trainer.refresh_count == 0
    second = trainer.step()
    assert second['refreshed'] is True
    assert trainer.refresh_count == 1
    assert trainer.last_refresh_step == 2
    assert trainer.trajectory_fingerprint == second['trajectory_fingerprint']
    assert trainer.trajectory.positions.shape[0] == 21


def test_online_checkpoint_resume_replays_refresh_and_rng_exactly(tmp_path):
    batch = _batch()
    uninterrupted = _trainer(batch, refresh_interval=2)
    for _ in range(3):
        uninterrupted.step()
    path = tmp_path / 'online.pt'
    uninterrupted.save_checkpoint(path)
    expected = [uninterrupted.step() for _ in range(4)]
    expected_state = {name: value.clone() for name, value in uninterrupted.model.state_dict().items()}
    expected_trajectory = uninterrupted.trajectory.positions.clone()
    expected_histogram = uninterrupted.k_histogram.clone()

    resumed = OnlineSelfStateTrainer.from_checkpoint(path, batch)
    actual = [resumed.step() for _ in range(4)]
    assert actual == expected
    assert torch.equal(resumed.k_histogram, expected_histogram)
    assert torch.equal(resumed.trajectory.positions, expected_trajectory)
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected_state[name])


def test_online_checkpoint_binds_current_trajectory(tmp_path):
    batch = _batch()
    trainer = _trainer(batch)
    trainer.step()
    path = tmp_path / 'online.pt'
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location='cpu')
    payload['trajectory_positions'] = payload['trajectory_positions'].clone()
    payload['trajectory_positions'][-1, 0, 0] += 0.001
    torch.save(payload, path)
    with pytest.raises(ValueError, match='trajectory fingerprint'):
        OnlineSelfStateTrainer.from_checkpoint(path, batch)


def test_online_trainer_rejects_bridge_rate():
    with pytest.raises(ValueError, match='physical remaining'):
        OnlineSelfStateTrainer(
            _batch(),
            model_config={'encoder_layers': 1, 'knn': 4,
                          'motion_parameterization': 'bridge_rate'},
        )
