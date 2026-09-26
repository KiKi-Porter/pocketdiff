import pickle
from pathlib import Path

import pytest
import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.diffusion import DiffusionMotionAdapter, diffusion_motion_loss, sample_diffusion_state

DATA_ROOT = Path('Apo2Mol-main/Apo2MOl-dataset/data_folder')
SPLIT_PATH = DATA_ROOT.parent / 'split_druglike_dict.pkl'


def _state():
    if not DATA_ROOT.is_dir() or not SPLIT_PATH.is_file():
        pytest.skip('Apo2Mol raw dataset unavailable')
    with SPLIT_PATH.open('rb') as handle:
        record = pickle.load(handle)['train'][0]
    value = Apo2MolAdapter(DATA_ROOT).convert_record(record)
    return sample_diffusion_state(value, 0.5, generator=torch.Generator().manual_seed(5))


def test_single_step_forward_backward_and_update():
    state = _state()
    model = DiffusionMotionAdapter(dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    pred = model(state.model_input)
    objective = diffusion_motion_loss(pred, state.target)
    assert torch.isfinite(objective.loss)
    objective.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optimizer.step()
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert changed


def test_checkpoint_reload_reproduces_output():
    state = _state()
    model = DiffusionMotionAdapter(dropout=0.0)
    model.eval()
    with torch.no_grad():
        expected = model(state.model_input)
    restored = DiffusionMotionAdapter(dropout=0.0)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    with torch.no_grad():
        actual = restored(state.model_input)
    assert torch.equal(expected.translation_local, actual.translation_local)
    assert torch.equal(expected.rotation_local, actual.rotation_local)
    assert torch.equal(expected.chi, actual.chi)
