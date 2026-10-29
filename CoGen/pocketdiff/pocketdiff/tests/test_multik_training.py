from dataclasses import replace

import pytest
import torch

from pocketdiff.tests.test_training import _batch
from pocketdiff.training import MultiKCleanTrainer, evaluate_multik_clean


CONFIG = {'encoder_layers': 1, 'knn': 4}


def test_training_resamples_all_times_and_reduces_fixed_grid_loss():
    batch = _batch()
    before = {name: value.clone() for name, value in batch.__dict__.items() if isinstance(value, torch.Tensor)}
    trainer = MultiKCleanTrainer(batch, model_config=CONFIG, seed=7)
    initial = evaluate_multik_clean(trainer.model, batch)
    records = [trainer.step() for _ in range(120)]
    final = evaluate_multik_clean(trainer.model, batch)
    assert final['mean']['loss'] < initial['mean']['loss']
    assert all(int(n) > 0 for n in trainer.k_histogram)
    assert trainer.k_histogram.sum() == 240
    expected_generator = torch.Generator().manual_seed(8)
    for record in records:
        assert record['pocket_k'] == torch.randint(20, (2,), generator=expected_generator).tolist()
        assert record['gradient_norm_before_clip'] > 0
    for name, value in before.items():
        assert torch.equal(getattr(batch, name), value), name


def test_eval_is_deterministic_preserves_mode_rng_and_uses_each_time():
    batch = _batch()
    trainer = MultiKCleanTrainer(batch, model_config=CONFIG)
    trainer.model.train()
    rng = torch.get_rng_state().clone()
    first = evaluate_multik_clean(trainer.model, batch)
    second = evaluate_multik_clean(trainer.model, batch)
    assert first == second
    assert trainer.model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert [r['k'] for r in first['per_k']] == list(range(20))
    assert len(first['per_graph']) == 40
    # Zero initialized model leaves apo coordinates unchanged at k=0.
    assert first['per_k'][0]['endpoint_holo_rmsd'] == first['per_k'][0]['current_holo_rmsd']


def test_checkpoint_resume_replays_optimizer_dropout_and_k_exactly(tmp_path):
    batch = _batch()
    trainer = MultiKCleanTrainer(batch, model_config=CONFIG, seed=32)
    for _ in range(4):
        trainer.step()
    path = tmp_path / 'resume.pt'
    trainer.save_checkpoint(path, metadata={'role': 'test'})
    expected_records = [trainer.step() for _ in range(4)]
    expected = {n: v.clone() for n, v in trainer.model.state_dict().items()}
    resumed = MultiKCleanTrainer.from_checkpoint(path, batch)
    assert [resumed.step() for _ in range(4)] == expected_records
    assert resumed.step_count == 8
    assert torch.equal(resumed.k_histogram, trainer.k_histogram)
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_resume_rejects_changed_data_or_geometry_version(tmp_path):
    batch = _batch()
    trainer = MultiKCleanTrainer(batch, model_config=CONFIG)
    path = tmp_path / 'resume.pt'
    trainer.save_checkpoint(path)
    with pytest.raises(ValueError, match='data or sample order'):
        MultiKCleanTrainer.from_checkpoint(path, replace(batch, ligand_pos=batch.ligand_pos + 1))
    payload = torch.load(path)
    payload['geometry_version'] = 'geometry-v1'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='geometry version'):
        MultiKCleanTrainer.from_checkpoint(path, batch)


def test_nonfinite_gradient_raises_before_optimizer_update():
    trainer = MultiKCleanTrainer(_batch(), model_config=CONFIG)
    before = {n: v.clone() for n, v in trainer.model.state_dict().items()}
    hook = trainer.model.motion_head.network[-1].weight.register_hook(lambda grad: grad * float('nan'))
    try:
        with pytest.raises(RuntimeError, match='non-finite'):
            trainer.step()
    finally:
        hook.remove()
    assert trainer.step_count == 0
    assert trainer.k_histogram.sum() == 0
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize('lr', [0., -1., float('nan')])
def test_invalid_learning_rate_rejected(lr):
    with pytest.raises(ValueError, match='learning_rate'):
        MultiKCleanTrainer(_batch(), learning_rate=lr)
