from dataclasses import replace

import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.geometry import (build_residue_frames, remaining_transform_current_to_holo,
                                periodic_chi_delta, oracle_rigid_chi_reconstruction, so3_exp)
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.joint_update import apply_joint_update
from pocketdiff.rollout import rollout_joint_from_apo
from pocketdiff.tests.test_current_joint_model import inputs, model
from pocketdiff.training import build_joint_reference


def example():
    state = replace(inputs(), pocket_k=torch.zeros(2, dtype=torch.long),
                    targetdiff_t=torch.full((2,), 199, dtype=torch.long))
    holo = state.protein_pos.clone()
    # CYS chi1 turns 90 degrees without changing bond lengths.
    holo[[5, 11], 1] -= 1.
    holo[[5, 11], 2] += 1.
    holo = holo @ so3_exp(torch.tensor([.1, -.2, .3])).T + .2
    return state, holo


def oracle_step(state, holo):
    ids, names = state.atom_to_residue_global, state.protein_atom_name
    nr = state.residue_type.numel()
    frames = build_residue_frames(state.protein_pos, ids, names, num_residues=nr)
    target = build_residue_frames(holo, ids, names, num_residues=nr)
    remaining = remaining_transform_current_to_holo(frames.origins, frames.frames,
                                                   target.origins, target.frames)
    current_chi = build_current_chi_state(state.protein_pos, names, ids, ['CYS'] * nr)
    target_chi = build_current_chi_state(holo, names, ids, ['CYS'] * nr)
    chi = periodic_chi_delta(current_chi.angles, target_chi.angles,
                             current_chi.geometry_rotatable_mask & target_chi.geometry_rotatable_mask)
    pred = PocketDiffPrediction(remaining.translation_local, remaining.rotvec_local,
                               chi, remaining.valid, {})
    return apply_joint_update(state.protein_pos, state.apo_pos_ref, ids, names, ['CYS'] * nr,
                              pred, remaining_steps=(20 - state.pocket_k)[state.batch_residue])


def test_reference_endpoints_and_oracle_match_each_of_twenty_steps():
    state, holo = example()
    reference = build_joint_reference(state, holo.requires_grad_())
    assert reference.positions.shape == (21, 12, 3)
    assert not reference.positions.requires_grad
    assert torch.equal(reference.positions[0], state.apo_pos_ref)
    endpoint = oracle_rigid_chi_reconstruction(state.protein_pos, holo,
        state.atom_to_residue_global, state.protein_atom_name, ['CYS'] * 2,
        torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(reference.positions[-1], endpoint.rigid_chi_positions, atol=3e-6, rtol=0)
    for k in range(20):
        batch = reference.batch_at(torch.full((2,), k, dtype=torch.long))
        expected = oracle_step(batch.inputs, holo).protein_pos_next
        torch.testing.assert_close(expected, batch.target_pos_next, atol=3e-6, rtol=0)


def test_mixed_k_selects_per_graph_current_and_next_with_label_free_kwargs():
    state, holo = example()
    ref = build_joint_reference(state, holo)
    batch = ref.batch_at(torch.tensor([0, 19]))
    assert torch.equal(batch.inputs.protein_pos[:6], state.protein_pos[:6])
    assert torch.equal(batch.inputs.protein_pos[6:], ref.positions[19, 6:])
    assert torch.equal(batch.target_pos_next[6:], ref.positions[20, 6:])
    assert batch.inputs.targetdiff_t.tolist() == [199, 9]
    assert not {'protein_pos_holo', 'supervision_frame_valid', 'chi_mask'} & batch.inputs.model_kwargs().keys()
    assert torch.equal(ref.apo_inputs.protein_pos, state.apo_pos_ref)


@pytest.mark.parametrize('k', [torch.tensor([-1, 0]), torch.tensor([0, 20]),
                             torch.tensor([0]), torch.tensor([0., 1.])])
def test_reference_rejects_invalid_k(k):
    state, holo = example()
    with pytest.raises(ValueError):
        build_joint_reference(state, holo).batch_at(k)


def test_reference_rejects_nonapo_state_or_bad_holo():
    state, holo = example()
    with pytest.raises(ValueError, match='apo'):
        build_joint_reference(replace(state, protein_pos=state.protein_pos + .1), holo)
    with pytest.raises(ValueError, match='holo'):
        build_joint_reference(state, torch.full_like(holo, float('nan')))


def test_holo_frame_mask_is_supervision_only_and_endpoint_keeps_apo_geometry():
    state, holo = example()
    holo[0] = holo[1]  # target N/CA degeneracy only in graph 0
    holo[9] += .4  # distort graph 1 oxygen without changing its N/CA/C frame
    reference = build_joint_reference(state, holo)
    assert reference.reference_frame_valid.tolist() == [False, True]
    assert reference.supervision_frame_valid.tolist() == [False, True]
    batch = reference.batch_at(torch.tensor([0, 19]))
    assert batch.inputs.model_kwargs()['frame_valid'].tolist() == [True, True]
    assert torch.equal(reference.positions[:, :6], state.apo_pos_ref[:6].expand(21, -1, -1))
    assert not torch.allclose(reference.positions[-1, 9], holo[9])
    original_length = (state.apo_pos_ref[9] - state.apo_pos_ref[8]).norm()
    final_length = (reference.positions[-1, 9] - reference.positions[-1, 8]).norm()
    torch.testing.assert_close(final_length, original_length)


def test_zero_rollout_stays_bitwise_apo_for_twenty_steps():
    state, _ = example()
    network = model().train()
    before = {key: val.clone() for key, val in network.state_dict().items()}
    rng = torch.get_rng_state()
    trajectory = rollout_joint_from_apo(network, state)
    assert torch.equal(trajectory.positions, state.apo_pos_ref.expand(21, -1, -1))
    assert network.training and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(val, before[key]) for key, val in network.state_dict().items())


def test_nonzero_rollout_uses_previous_prediction_and_preserves_inputs():
    state, _ = example()
    network = model(nonzero=True)
    before = {key: val.clone() for key, val in vars(state).items() if isinstance(val, torch.Tensor)}
    seen = []
    def capture(module, args, kwargs):
        seen.append((kwargs['protein_pos'].clone(), kwargs['pocket_k'].clone()))
    # Torch 1.13 lacks forward_pre_hook(with_kwargs); wrap forward temporarily.
    original = network.forward
    def forward(**kwargs):
        capture(network, (), kwargs)
        return original(**kwargs)
    network.forward = forward
    try:
        trajectory = rollout_joint_from_apo(network, state)
    finally:
        network.forward = original
    assert len(seen) == 20
    for k, (pos, step) in enumerate(seen):
        assert torch.equal(pos, trajectory.positions[k])
        assert step.tolist() == [k, k]
    assert not torch.equal(trajectory.positions[-1], state.apo_pos_ref)
    assert all(torch.equal(getattr(state, key), val) for key, val in before.items())
    assert not network.training


def test_rollout_rejects_teacher_forced_start():
    state, _ = example()
    with pytest.raises(ValueError, match='apo'):
        rollout_joint_from_apo(model(), replace(state, protein_pos=state.protein_pos + .1))


def test_rollout_restores_mode_after_forward_failure(monkeypatch):
    state, _ = example()
    network = model().train()
    def fail(**kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(network, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        rollout_joint_from_apo(network, state)
    assert network.training
