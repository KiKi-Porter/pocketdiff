import torch

from pocketdiff.training.diffusion import DiffusionTrainConfig, train_diffusion_model


def test_batched_accumulated_training_records_graph_protocol():
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    first = _synthetic_complex()
    second = _synthetic_complex()
    second.sample_id = "synthetic-2"
    model, optimizer, rows = train_diffusion_model(
        [first, second],
        DiffusionTrainConfig(
            train_count=2,
            holdout_count=0,
            updates=1,
            times=(0.25, 0.75),
            batch_size=2,
            gradient_accumulation_steps=2,
            self_state_probability=1.0,
            self_state_steps=1,
            self_state_rollout_steps=2,
            endpoint_weight=0.0,
        ),
    )
    assert len(rows) == 1
    assert rows[0]["graph_count"] == 4
    assert rows[0]["self_state_count"] == 4
    assert rows[0]["finite"] is True
    assert any(value.grad is None for value in model.parameters()) or all(
        torch.isfinite(value).all() for value in model.parameters()
    )
