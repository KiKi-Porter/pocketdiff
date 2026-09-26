import pytest
import torch

from pocketdiff.tests.test_training import _batch
from pocketdiff.training import SelfStateTrainer, evaluate_self_state


def _trajectory(batch):
    # Use a finite detached trajectory with valid frames. The endpoint state
    # remains apo, while labels are recomputed from each selected state.
    return batch.protein_pos.expand(21, -1, -1).clone()


def test_self_state_training_step_uses_random_k_and_finite_gradient():
    batch = _batch()
    trainer = SelfStateTrainer(batch, _trajectory(batch), model_config={'encoder_layers': 1, 'knn': 4}, seed=7)
    records = [trainer.step() for _ in range(25)]
    assert all(record['loss_parameterization'] == 'remaining' for record in records)
    assert all(record['gradient_norm_before_clip'] > 0 for record in records)
    assert int(trainer.k_histogram.sum()) == 50
    assert all(torch.isfinite(value).all() for value in trainer.model.parameters())


def test_self_state_evaluation_is_deterministic_and_restores_mode():
    batch = _batch()
    trajectory = _trajectory(batch)
    trainer = SelfStateTrainer(batch, trajectory, model_config={'encoder_layers': 1, 'knn': 4})
    trainer.model.train()
    rng = torch.get_rng_state().clone()
    first = evaluate_self_state(trainer.model, batch, trajectory)
    second = evaluate_self_state(trainer.model, batch, trajectory)
    assert first == second
    assert trainer.model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert [row['k'] for row in first['per_k']] == list(range(20))


def test_self_state_checkpoint_resume_is_exact(tmp_path):
    batch = _batch()
    trajectory = _trajectory(batch)
    trainer = SelfStateTrainer(batch, trajectory, model_config={'encoder_layers': 1, 'knn': 4}, seed=32)
    for _ in range(4):
        trainer.step()
    path = tmp_path/'resume.pt'
    trainer.save_checkpoint(path)
    expected_records = [trainer.step() for _ in range(4)]
    expected = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    resumed = SelfStateTrainer.from_checkpoint(path, batch, trajectory)
    assert [resumed.step() for _ in range(4)] == expected_records
    assert torch.equal(resumed.k_histogram, trainer.k_histogram)
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_self_state_checkpoint_rejects_changed_trajectory(tmp_path):
    batch = _batch()
    trajectory = _trajectory(batch)
    trainer = SelfStateTrainer(batch, trajectory, model_config={'encoder_layers': 1, 'knn': 4})
    path = tmp_path/'resume.pt'
    trainer.save_checkpoint(path)
    changed = trajectory.clone()
    changed[-1, 0, 0] += 0.001
    with pytest.raises(ValueError, match='trajectory'):
        SelfStateTrainer.from_checkpoint(path, batch, changed)


def test_self_state_trainer_rejects_bridge_rate_mode():
    with pytest.raises(ValueError, match='physical remaining'):
        SelfStateTrainer(_batch(), _trajectory(_batch()),
                         model_config={'encoder_layers': 1, 'knn': 4,
                                       'motion_parameterization': 'bridge_rate'})
