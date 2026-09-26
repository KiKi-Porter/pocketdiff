import pytest
import torch

from pocketdiff_v4.batching import collate_complexes
from pocketdiff_v4.evaluate import _sample_metrics
from pocketdiff_v4.sampler import (
    initial_state_from_apo,
    sample_complexes,
    sample_complexes_with_trajectory,
    schedule_fraction,
)
from pocketdiff_v4.tests.test_rollout_objective import _model, _record


def test_sampler_requires_only_input_and_is_sample_seed_deterministic():
    batch = collate_complexes([_record("deterministic")])
    inputs = batch["input"]
    model = _model().eval()
    first = sample_complexes(
        model,
        inputs,
        ["deterministic"],
        steps=3,
        seed=71,
        initial_noise_scale=1.0,
    )
    second = sample_complexes(
        model,
        inputs,
        ["deterministic"],
        steps=3,
        seed=71,
        initial_noise_scale=1.0,
    )
    other_seed = sample_complexes(
        model,
        inputs,
        ["deterministic"],
        steps=3,
        seed=72,
        initial_noise_scale=1.0,
    )
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.allclose(first, other_seed)


def test_zero_motion_scale_returns_apo_seeded_initial_state():
    batch = collate_complexes([_record("zero-motion")])
    inputs = batch["input"]
    model = _model().eval()
    expected = initial_state_from_apo(
        inputs, ["zero-motion"], seed=71, initial_noise_scale=1.0
    )
    actual = sample_complexes(
        model,
        inputs,
        ["zero-motion"],
        steps=4,
        seed=71,
        motion_scale=0.0,
        initial_noise_scale=1.0,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_zero_motion_scale_does_not_call_model_and_preserves_trajectory_length():
    class ExplodingModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("model should not run when motion_scale=0")

    batch = collate_complexes([_record("zero-motion-short-circuit")])
    inputs = batch["input"]
    predicted, trajectory = sample_complexes_with_trajectory(
        ExplodingModel(),
        inputs,
        ["zero-motion-short-circuit"],
        steps=4,
        seed=71,
        motion_scale=0.0,
        initial_noise_scale=0.0,
    )
    assert torch.allclose(predicted, inputs["apo_pos"])
    assert len(trajectory) == 5
    assert all(torch.allclose(state, inputs["apo_pos"]) for state in trajectory)


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


def test_sampler_defaults_to_clean_apo_initial_state():
    batch = collate_complexes([_record("default-clean")])
    inputs = batch["input"]
    expected = sample_complexes(
        _model().eval(),
        inputs,
        ["default-clean"],
        steps=4,
        seed=71,
        motion_scale=0.0,
    )
    assert torch.allclose(expected, inputs["apo_pos"], atol=1e-6, rtol=1e-6)


def test_schedule_fraction_variants():
    assert schedule_fraction(4, 4) == pytest.approx(0.25)
    assert schedule_fraction(1, 4) == pytest.approx(1.0)
    assert schedule_fraction(
        1, 4, schedule_type="clipped", max_fraction=0.25
    ) == pytest.approx(0.25)
    assert schedule_fraction(
        1, 4, schedule_type="damped", damping=0.25
    ) == pytest.approx(0.5)
    assert schedule_fraction(
        3, 4, schedule_type="fixed", fixed_fraction=0.2
    ) == pytest.approx(0.2)


def test_sampler_rejects_invalid_schedule_configuration():
    with pytest.raises(ValueError, match="max_fraction"):
        schedule_fraction(1, 4, schedule_type="clipped")
    with pytest.raises(ValueError, match="fixed_fraction"):
        schedule_fraction(1, 4, schedule_type="fixed")
    with pytest.raises(ValueError, match="unknown schedule_type"):
        schedule_fraction(1, 4, schedule_type="bad")


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
    assert metrics["sample_holo_ca_rmsd"] >= 0
    assert metrics["sample_holo_backbone_rmsd"] >= 0
    assert metrics["rigid_oracle_holo_backbone_rmsd"] >= 0
