from dataclasses import replace
import pickle
from pathlib import Path

import pytest
import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.diffusion import (
    DiffusionMotionAdapter,
    DiffusionMotionPrediction,
    diffusion_endpoint_loss,
    diffusion_motion_loss,
    predict_holo_coordinates,
    sample_diffusion_state,
)


DATA_ROOT = Path("Apo2Mol-main/Apo2MOl-dataset/data_folder")
SPLIT_PATH = DATA_ROOT.parent / "split_druglike_dict.pkl"


def _state(t: float = 0.5):
    if not DATA_ROOT.is_dir() or not SPLIT_PATH.is_file():
        pytest.skip("Apo2Mol raw dataset unavailable")
    with SPLIT_PATH.open("rb") as handle:
        record = pickle.load(handle)["train"][0]
    value = Apo2MolAdapter(DATA_ROOT).convert_record(record)
    return sample_diffusion_state(
        value,
        t,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
        generator=torch.Generator().manual_seed(58),
    )


def _zero_prediction(state):
    return DiffusionMotionPrediction(
        translation_local=torch.zeros_like(state.target.translation_target_local),
        rotation_local=torch.zeros_like(state.target.rotation_target_local),
        chi=torch.zeros_like(state.target.chi_target),
    )


def test_zero_update_at_clean_holo_is_exact_and_endpoint_loss_is_zero():
    state = _state(0.0)
    prediction = _zero_prediction(state)
    endpoint = predict_holo_coordinates(state.model_input, prediction)

    assert torch.equal(endpoint, state.model_input.protein_pos)
    assert torch.allclose(endpoint, state.target.protein_pos_holo, atol=1.0e-6)
    loss = diffusion_endpoint_loss(prediction, state.target, state.model_input)
    assert torch.isfinite(loss)
    assert float(loss) <= 1.0e-12


def test_endpoint_loss_backpropagates_through_motion_prediction():
    state = _state(1.0)
    model = DiffusionMotionAdapter(dropout=0.0)
    prediction = model(state.model_input)
    objective = diffusion_motion_loss(
        prediction,
        state.target,
        model_input=state.model_input,
        endpoint_weight=0.5,
    )
    assert torch.isfinite(objective.loss)
    assert torch.isfinite(objective.endpoint_loss)
    objective.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_endpoint_loss_expands_residue_mask_to_atoms():
    state = _state(0.5)
    active = torch.where(state.target.target_valid)[0]
    if active.numel() < 2:
        pytest.skip("need at least two valid residues for mask regression")
    excluded_residue = int(active[0])
    atom_mask = state.model_input.atom_to_residue == excluded_residue
    if not bool(atom_mask.any()):
        pytest.skip("selected valid residue has no atoms")

    prediction = _zero_prediction(state)
    masked_valid = state.target.target_valid.clone()
    masked_valid[excluded_residue] = False
    masked_target = replace(state.target, target_valid=masked_valid)

    perturbed_holo = state.target.protein_pos_holo.clone()
    perturbed_holo[atom_mask] += 100.0
    perturbed_target = replace(
        masked_target,
        protein_pos_holo=perturbed_holo,
    )
    reference_loss = diffusion_endpoint_loss(
        prediction,
        masked_target,
        state.model_input,
    )
    perturbed_loss = diffusion_endpoint_loss(
        prediction,
        perturbed_target,
        state.model_input,
    )
    assert torch.equal(reference_loss, perturbed_loss)


def test_legacy_motion_loss_keeps_zero_endpoint_term_without_input():
    state = _state(0.5)
    model = DiffusionMotionAdapter(dropout=0.0)
    prediction = model(state.model_input)
    objective = diffusion_motion_loss(prediction, state.target)
    assert torch.equal(objective.endpoint_loss, prediction.translation_local.sum() * 0.0)
    assert torch.equal(
        objective.loss,
        objective.translation_loss + objective.rotation_loss + objective.chi_loss,
    )
