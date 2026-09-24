import pytest
import torch

from pocketdiff_v4.batching import collate_complexes
from pocketdiff_v4.evaluate import _sample_metrics
from pocketdiff_v4.sampler import initial_state_from_apo, sample_complexes
from pocketdiff_v4.tests.test_rollout_objective import _model, _record


def test_sampler_requires_only_input_and_is_sample_seed_deterministic():
    batch = collate_complexes([_record("deterministic")])
    inputs = batch["input"]
    model = _model().eval()
    first = sample_complexes(
        model, inputs, ["deterministic"], steps=3, seed=71
    )
    second = sample_complexes(
        model, inputs, ["deterministic"], steps=3, seed=71
    )
    other_seed = sample_complexes(
        model, inputs, ["deterministic"], steps=3, seed=72
    )
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.allclose(first, other_seed)


def test_zero_motion_scale_returns_apo_seeded_initial_state():
    batch = collate_complexes([_record("zero-motion")])
    inputs = batch["input"]
    model = _model().eval()
    expected = initial_state_from_apo(inputs, ["zero-motion"], seed=71)
    actual = sample_complexes(
        model,
        inputs,
        ["zero-motion"],
        steps=4,
        seed=71,
        motion_scale=0.0,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_zero_initial_noise_starts_from_apo():
    batch = collate_complexes([_record("zero-noise")])
    inputs = batch["input"]
    actual = initial_state_from_apo(
        inputs,
        ["zero-noise"],
        seed=71,
        initial_noise_scale=0.0,
    )
    assert torch.allclose(actual, inputs["apo_pos"], atol=1e-6, rtol=1e-6)


def test_sampler_rejects_motion_scale_outside_unit_interval():
    batch = collate_complexes([_record("bad-scale")])
    with pytest.raises(ValueError, match="motion_scale"):
        sample_complexes(
            _model().eval(),
            batch["input"],
            ["bad-scale"],
            motion_scale=1.01,
        )


def test_sampler_rejects_initial_noise_scale_outside_unit_interval():
    batch = collate_complexes([_record("bad-noise-scale")])
    with pytest.raises(ValueError, match="noise_scale"):
        sample_complexes(
            _model().eval(),
            batch["input"],
            ["bad-noise-scale"],
            initial_noise_scale=1.01,
        )


def test_sample_metrics_use_holo_only_after_sampling():
    record = _record("scoring")
    inputs_only = collate_complexes([record])["input"]
    predicted = sample_complexes(
        _model().eval(), inputs_only, ["scoring"], steps=2, seed=17
    )
    metrics = _sample_metrics(record, predicted)
    assert metrics["sample_id"] == "scoring"
    assert metrics["finite"]
    assert metrics["pocket_atoms"] > 0
