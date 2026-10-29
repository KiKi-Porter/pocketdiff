import pickle
from pathlib import Path

import pytest
import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.diffusion.state import sample_diffusion_state

DATA_ROOT = Path('Apo2Mol-main/Apo2MOl-dataset/data_folder')
SPLIT_PATH = DATA_ROOT.parent / 'split_druglike_dict.pkl'


def _record():
    if not DATA_ROOT.is_dir() or not SPLIT_PATH.is_file():
        pytest.skip('Apo2Mol raw dataset unavailable')
    with SPLIT_PATH.open('rb') as handle:
        return pickle.load(handle)['train'][0]


def _complex():
    return Apo2MolAdapter(DATA_ROOT).convert_record(_record())


def test_diffusion_state_endpoints_and_shapes():
    value = _complex()
    state0 = sample_diffusion_state(value, 0.0)
    state1 = sample_diffusion_state(value, 1.0, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(state0.model_input.protein_pos, value.protein_pos_holo, atol=2e-5)
    assert torch.allclose(state0.target.protein_pos_holo, value.protein_pos_holo)
    assert state0.model_input.diffusion_time.tolist() == [0.0]
    assert state1.model_input.protein_pos.shape == value.protein_pos_apo.shape
    assert state1.target.translation_target_local.shape == (value.num_residues, 3)
    assert state1.target.rotation_target_local.shape == (value.num_residues, 3)
    assert state1.target.chi_target.shape == (value.num_residues, 5)
    assert bool(torch.isfinite(state1.model_input.protein_pos).all())
    assert bool(torch.isfinite(state1.target.translation_target_local).all())


def test_zero_noise_diffusion_endpoints_match_holo_and_apo():
    value = _complex()
    for time_value, reference in (
        (0.0, value.protein_pos_holo),
        (1.0, value.protein_pos_apo),
    ):
        state = sample_diffusion_state(
            value,
            time_value,
            translation_noise_scale=0.0,
            rotation_noise_scale=0.0,
            chi_noise_scale=0.0,
        )
        assert torch.equal(state.model_input.protein_pos, reference)


def test_diffusion_state_seed_reproducibility():
    value = _complex()
    a = sample_diffusion_state(value, 0.5, generator=torch.Generator().manual_seed(17))
    b = sample_diffusion_state(value, 0.5, generator=torch.Generator().manual_seed(17))
    assert torch.equal(a.model_input.protein_pos, b.model_input.protein_pos)
    assert torch.equal(a.noise_translation_local, b.noise_translation_local)
    assert torch.equal(a.target.rotation_target_local, b.target.rotation_target_local)


def test_diffusion_state_rejects_invalid_time():
    with pytest.raises(ValueError, match='\[0, 1\]'):
        sample_diffusion_state(_complex(), 1.1)
