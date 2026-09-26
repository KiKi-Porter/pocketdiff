import pytest
import torch

from pocketdiff.geometry.bridge import remaining_transform_current_to_holo
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.tests.test_training import _batch
from pocketdiff.training import build_bridge_batch, build_self_state_batch


def test_k0_self_state_matches_clean_and_mixed_graph_selects_own_state():
    clean = _batch()
    # Distinct per-step states make graph-specific selection observable.
    trajectory = torch.stack([clean.apo_pos_ref + float(step) * .01 for step in range(21)])
    selected = build_self_state_batch(clean, trajectory, torch.tensor([0, 19]))
    assert torch.equal(selected.protein_pos[:4], trajectory[0, :4])
    assert torch.equal(selected.protein_pos[4:], trajectory[19, 4:])
    assert selected.targetdiff_t.tolist() == [199, 9]
    clean0 = build_self_state_batch(clean, trajectory, torch.zeros(2, dtype=torch.long))
    bridge0 = build_bridge_batch(clean, torch.zeros(2, dtype=torch.long))
    assert torch.equal(clean0.protein_pos, clean.protein_pos)
    assert torch.allclose(clean0.target_translation_local, bridge0.target_translation_local, atol=1e-6)
    assert torch.allclose(clean0.target_rotvec_local, bridge0.target_rotvec_local, atol=1e-6)
    assert torch.equal(clean0.frame_valid, bridge0.frame_valid)


def test_labels_equal_independent_current_to_holo_geometry_and_inputs_unchanged():
    clean = _batch()
    before = {name: value.clone() for name, value in clean.__dict__.items() if isinstance(value, torch.Tensor)}
    trajectory = torch.stack([clean.apo_pos_ref + float(step) * .002 for step in range(21)])
    state = build_self_state_batch(clean, trajectory, torch.tensor([5, 12]))
    frames = build_residue_frames(state.protein_pos, state.atom_to_residue_global,
                                  state.protein_atom_name, num_residues=state.residue_type.numel())
    holo = build_residue_frames(clean.protein_pos_holo, clean.atom_to_residue_global,
                                clean.protein_atom_name, num_residues=clean.residue_type.numel())
    expected = remaining_transform_current_to_holo(frames.origins, frames.frames,
                                                   holo.origins, holo.frames,
                                                   frame_valid=state.frame_valid)
    assert torch.equal(state.target_translation_local, expected.translation_local)
    assert torch.equal(state.target_rotvec_local, expected.rotvec_local)
    assert not state.protein_pos.requires_grad
    assert not state.target_translation_local.requires_grad
    for name, value in before.items():
        assert torch.equal(getattr(clean, name), value), name


@pytest.mark.parametrize('bad', [
    'short', 'nonfinite', 'wrong_step0', 'wrong_k', 'wrong_shape',
])
def test_invalid_self_state_inputs_are_rejected(bad):
    clean = _batch()
    trajectory = torch.stack([clean.apo_pos_ref for _ in range(21)])
    k = torch.tensor([0, 1])
    if bad == 'short': trajectory = trajectory[:19]
    if bad == 'nonfinite': trajectory[3, 0, 0] = float('nan')
    if bad == 'wrong_step0': trajectory[0, 0, 0] += .1
    if bad == 'wrong_k': k[0] = 20
    if bad == 'wrong_shape': trajectory = trajectory[:, :-1]
    with pytest.raises(ValueError):
        build_self_state_batch(clean, trajectory, k)


def test_self_state_keeps_model_kwargs_free_of_holo_and_targets():
    clean = _batch()
    trajectory = torch.stack([clean.apo_pos_ref for _ in range(21)])
    state = build_self_state_batch(clean, trajectory, torch.tensor([1, 2]))
    kwargs = state.model_kwargs()
    assert 'protein_pos_holo' not in kwargs
    assert 'target_translation_local' not in kwargs
    assert 'target_rotvec_local' not in kwargs
    assert 'protein_pos_next_target' not in kwargs
