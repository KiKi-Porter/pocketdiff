import torch

from pocketdiff.diffusion.model import _backbone_progress_loss, _chain_continuity_loss
from pocketdiff.diffusion.state import sample_diffusion_state
from pocketdiff.tests.test_velocity_contract import _complex


def test_directed_backbone_progress_is_zero_for_oracle_bridge_step():
    value = _complex(angle=0.35, shift=(0.4, -0.2, 0.3))
    state = sample_diffusion_state(
        value,
        1.0,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
        target_parameterization="bridge_velocity",
        target_steps=20,
    )
    direction, magnitude = _backbone_progress_loss(
        state.target.next_protein_pos,
        state.target.next_protein_pos,
        state.model_input,
        state.target,
        threshold=0.05,
    )
    assert float(direction) < 1.0e-5
    assert float(magnitude) < 1.0e-5


def test_directed_backbone_progress_penalizes_wrong_and_zero_motion():
    value = _complex(angle=0.35, shift=(0.4, -0.2, 0.3))
    state = sample_diffusion_state(
        value,
        1.0,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
        target_parameterization="bridge_velocity",
        target_steps=20,
    )
    zero = torch.zeros_like(state.model_input.protein_pos)
    direction, magnitude = _backbone_progress_loss(
        zero,
        state.target.next_protein_pos,
        state.model_input,
        state.target,
        threshold=0.05,
    )
    assert float(direction) > 0.9
    assert float(magnitude) > 0.0


def test_chain_geometry_loss_is_zero_for_holo_and_penalizes_angle_distortion():
    value = _complex(angle=0.35, shift=(0.4, -0.2, 0.3))
    state = sample_diffusion_state(
        value,
        1.0,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
        target_parameterization="bridge_velocity",
        target_steps=20,
    )
    holo_loss = _chain_continuity_loss(
        state.target.protein_pos_holo,
        state.target.protein_pos_holo,
        state.model_input,
    )
    distorted = state.target.protein_pos_holo.clone()
    ca_indices = [
        index
        for index, name in enumerate(state.model_input.protein_atom_name)
        if name == "CA"
    ]
    assert ca_indices
    distorted[ca_indices[0], 2] += 0.75
    distorted.requires_grad_(True)
    distorted_loss = _chain_continuity_loss(
        distorted,
        state.target.protein_pos_holo,
        state.model_input,
    )
    assert float(holo_loss) < 1.0e-8
    assert float(distorted_loss) > 1.0e-5
    distorted_loss.backward()
    assert distorted.grad is not None
    assert bool(torch.isfinite(distorted.grad).all())
