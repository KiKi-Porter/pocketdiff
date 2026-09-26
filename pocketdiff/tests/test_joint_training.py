import math
from dataclasses import replace

import pytest
import torch

from pocketdiff.models import PocketDiffModel
from pocketdiff.tests.test_training import _batch
from pocketdiff.training import (
    evaluate_joint_clean,
    masked_joint_clean_loss,
    save_joint_checkpoint,
    train_joint_clean_batch,
)


def _chi_batch():
    batch = _batch()
    return replace(
        batch,
        chi_apo=torch.tensor([
            [math.pi - 0.1, 0.2, 0., 0., 0.],
            [0.1, -0.3, 0., 0., 0.],
        ]),
        chi_holo=torch.tensor([
            [-math.pi + 0.1, 0.4, 0., 0., 0.],
            [0.3, -0.3, 0., 0., 0.],
        ]),
        chi_mask=torch.tensor([
            [True, True, False, False, False],
            [True, True, False, False, False],
        ]),
    )


def test_joint_loss_requires_chi_head_and_has_finite_components():
    batch = _chi_batch()
    model = PocketDiffModel(encoder_layers=1, knn=4, predict_chi=True)
    prediction = model(**batch.model_kwargs())
    loss = masked_joint_clean_loss(prediction, batch)
    assert loss.valid_residue_count == 2
    assert loss.valid_chi_count == 4
    assert torch.isfinite(loss.loss)
    assert torch.isfinite(loss.motion_loss)
    assert torch.isfinite(loss.chi_loss)
    with pytest.raises(ValueError, match="remaining_chi"):
        masked_joint_clean_loss(
            PocketDiffModel(encoder_layers=1, knn=4)(**batch.model_kwargs()), batch
        )


def test_joint_loss_gradient_reaches_both_heads():
    torch.manual_seed(19)
    batch = _chi_batch()
    model = PocketDiffModel(encoder_layers=1, knn=4, predict_chi=True)
    prediction = model(**batch.model_kwargs())
    loss = masked_joint_clean_loss(prediction, batch)
    loss.loss.backward()
    assert model.motion_head.network[-1].weight.grad is not None
    assert model.chi_head.network[-1].weight.grad is not None
    assert torch.isfinite(model.motion_head.network[-1].weight.grad).all()
    assert torch.isfinite(model.chi_head.network[-1].weight.grad).all()
    assert float(model.chi_head.network[-1].weight.grad.abs().sum()) > 0.0


def test_joint_clean_trainer_reduces_loss_and_evaluator_restores_mode():
    torch.manual_seed(23)
    batch = _chi_batch()
    model = PocketDiffModel(encoder_layers=1, knn=4, predict_chi=True)
    initial = evaluate_joint_clean(model, batch)
    assert model.training is True
    records = train_joint_clean_batch(
        model, batch, steps=12, learning_rate=3e-3, log_every=6
    )
    final = evaluate_joint_clean(model, batch)
    assert records[-1].step == 12
    assert final.loss.item() < initial.loss.item()
    assert final.chi_loss.item() < initial.chi_loss.item()


def test_joint_clean_trainer_rejects_empty_chi_supervision():
    batch = _batch()
    model = PocketDiffModel(encoder_layers=1, knn=4, predict_chi=True)
    with pytest.raises(ValueError, match="no valid chi"):
        train_joint_clean_batch(model, batch, steps=2)


def test_joint_checkpoint_reload_preserves_both_heads(tmp_path):
    torch.manual_seed(31)
    batch = _chi_batch()
    model = PocketDiffModel(encoder_layers=1, knn=4, predict_chi=True)
    records = train_joint_clean_batch(model, batch, steps=2, log_every=1)
    path = tmp_path / "joint.pt"
    save_joint_checkpoint(
        path,
        model,
        config={"encoder_layers": 1, "knn": 4, "predict_chi": True},
        sample_ids=batch.sample_ids,
        final_record=records[-1],
    )
    payload = torch.load(path, map_location="cpu")
    restored = PocketDiffModel(**payload["model_config"])
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    restored.eval()
    with torch.no_grad():
        original = model(**batch.model_kwargs())
        reloaded = restored(**batch.model_kwargs())
    assert torch.allclose(original.remaining_translation_local, reloaded.remaining_translation_local)
    assert torch.allclose(original.remaining_rotvec_local, reloaded.remaining_rotvec_local)
    assert torch.allclose(original.remaining_chi, reloaded.remaining_chi)
