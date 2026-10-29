from dataclasses import replace

import torch

from pocketdiff.diffusion import DiffusionMotionAdapter, sample_diffusion_state
from pocketdiff.models.dynamicbind_vector import DynamicBindVectorBlock
from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex
from pocketdiff.geometry.so3 import so3_exp


def test_dynamicbind_vector_block_is_finite_and_se3_equivariant():
    torch.manual_seed(67)
    block = DynamicBindVectorBlock(hidden_dim=8, time_dim=4, vector_channels=3)
    residue_hidden = torch.randn(4, 8)
    origins = torch.randn(4, 3)
    ligand_hidden = torch.randn(3, 8)
    ligand_pos = torch.randn(3, 3)
    time_hidden = torch.randn(1, 4)
    valid = torch.ones(4, dtype=torch.bool)
    hidden, vector, contact = block(
        residue_hidden,
        origins,
        ligand_hidden,
        ligand_pos,
        time_hidden,
        valid,
    )

    rotation = so3_exp(torch.tensor([0.2, -0.3, 0.4]))
    shift = torch.tensor([2.0, -0.5, 0.7])
    moved_hidden, moved_vector, moved_contact = block(
        residue_hidden,
        origins @ rotation.transpose(0, 1) + shift,
        ligand_hidden,
        ligand_pos @ rotation.transpose(0, 1) + shift,
        time_hidden,
        valid,
    )
    assert hidden.shape == (4, 8)
    assert vector.shape == (4, 3)
    assert contact.shape == (4, 1)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(vector).all()
    assert torch.isfinite(contact).all()
    torch.testing.assert_close(hidden, moved_hidden, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        vector @ rotation.transpose(0, 1),
        moved_vector,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(contact, moved_contact, atol=2e-5, rtol=2e-5)


def test_dynamicbind_backend_forward_is_finite_and_local_invariant():
    value = _synthetic_complex()
    state = sample_diffusion_state(
        value,
        0.5,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
    )
    model = DiffusionMotionAdapter(encoder_backend="dynamicbind", dropout=0.0).eval()
    with torch.no_grad():
        first = model(state.model_input)
    rotation = so3_exp(torch.tensor([0.2, -0.3, 0.4]))
    shift = torch.tensor([2.0, -0.5, 0.7])
    moved = replace(
        state.model_input,
        protein_pos=state.model_input.protein_pos @ rotation.transpose(0, 1) + shift,
        apo_pos_ref=state.model_input.apo_pos_ref @ rotation.transpose(0, 1) + shift,
        ligand_pos=state.model_input.ligand_pos @ rotation.transpose(0, 1) + shift,
    )
    with torch.no_grad():
        second = model(moved)
    assert torch.isfinite(first.translation_local).all()
    assert torch.isfinite(first.rotation_local).all()
    assert torch.isfinite(first.chi).all()
    torch.testing.assert_close(
        first.translation_local, second.translation_local, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        first.rotation_local, second.rotation_local, atol=2e-5, rtol=2e-5
    )
