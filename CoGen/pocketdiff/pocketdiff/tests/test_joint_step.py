from dataclasses import replace
import math

import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.geometry.joint_update import apply_joint_update
from pocketdiff.geometry import periodic_chi_delta, so3_exp, oracle_rigid_chi_reconstruction
from pocketdiff.inference import predict_joint_step
from pocketdiff.tests.test_current_joint_model import inputs, model


def prediction(tr=None, rot=None, chi=None, valid=None):
    return PocketDiffPrediction(
        remaining_translation_local=torch.zeros(2, 3) if tr is None else tr,
        remaining_rotvec_local=torch.zeros(2, 3) if rot is None else rot,
        remaining_chi=torch.zeros(2, 5) if chi is None else chi,
        frame_valid=torch.ones(2, dtype=torch.bool) if valid is None else valid,
        diagnostics={},
    )


def apply(state, pred, steps=1, residues=None):
    return apply_joint_update(state.protein_pos, state.apo_pos_ref,
                              state.atom_to_residue_global, state.protein_atom_name,
                              ['CYS', 'CYS'] if residues is None else residues,
                              pred, remaining_steps=steps)


def test_zero_init_stationary_and_coordinate_loss_reaches_all_three_heads():
    state, network = inputs(), model()
    before = {k: v.clone() for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
    out = predict_joint_step(network, state)
    assert torch.equal(out.protein_pos_next, state.protein_pos)
    target = state.protein_pos.clone() + torch.tensor([.1, -.2, .3])
    target[[5, 11], 2] += .7
    (out.protein_pos_next - target).square().mean().backward()
    rigid_grad = network.motion_head.network[-1].weight.grad
    chi_grad = network.current_chi_head.network[-1].weight.grad
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
    for grad in (rigid_grad[:3], rigid_grad[3:], chi_grad):
        assert grad.abs().sum() > 1e-5
    assert all(torch.equal(getattr(state, k), v) for k, v in before.items())


def test_so3_zero_derivative_is_analytic_and_finite():
    v = torch.zeros(1, 3, requires_grad=True)
    so3_exp(v)[0, 1, 2].backward()
    torch.testing.assert_close(v.grad, torch.tensor([[-1., 0., 0.]]))


@pytest.mark.parametrize('amplitude', [0., .17])
def test_joint_coordinates_match_finite_difference_gradients(amplitude):
    state = inputs()
    parameters = torch.full((2, 11), amplitude, requires_grad=True)
    weights = torch.randn(12, 3, generator=torch.Generator().manual_seed(32))
    def function(v):
        return (apply(state, prediction(v[:, :3], v[:, 3:6], v[:, 6:]), steps=2).protein_pos_next * weights).sum()
    function(parameters).backward()
    assert torch.isfinite(parameters.grad).all()
    epsilon = .002
    for col in range(7):  # three translations, three rotations, CYS chi1
        plus, minus = parameters.detach().clone(), parameters.detach().clone()
        plus[0, col] += epsilon
        minus[0, col] -= epsilon
        expected = (function(plus) - function(minus)) / (2 * epsilon)
        torch.testing.assert_close(parameters.grad[0, col], expected, atol=.003, rtol=.01)


def test_per_graph_fraction_and_readback_of_next_state():
    state = inputs()
    out = predict_joint_step(model(nonzero=True), state)
    n = torch.tensor([[20.], [1.]])
    torch.testing.assert_close(out.applied_translation_local, out.prediction.remaining_translation_local / n)
    torch.testing.assert_close(out.applied_rotvec_local, out.prediction.remaining_rotvec_local / n)
    torch.testing.assert_close(out.applied_chi, out.prediction.remaining_chi / n)
    observed = periodic_chi_delta(out.diagnostics['current_chi'], out.diagnostics['chi_next'],
                                  out.diagnostics['chi_rotatable_mask'])
    torch.testing.assert_close(observed, out.applied_chi, atol=2e-6, rtol=0)


def test_nonzero_network_and_solver_are_se3_equivariant():
    state, network = inputs(), model(nonzero=True)
    rot = so3_exp(torch.tensor([.2, -.3, .4]))
    shift = torch.tensor([1.5, -.7, .2])
    moved = replace(state, protein_pos=state.protein_pos @ rot.T + shift,
                    apo_pos_ref=state.apo_pos_ref @ rot.T + shift,
                    ligand_pos=state.ligand_pos @ rot.T + shift)
    expected = predict_joint_step(network, state).protein_pos_next @ rot.T + shift
    actual = predict_joint_step(network, moved).protein_pos_next
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-6)


def test_invalid_frame_freezes_motion_and_missing_chi_cannot_move_atoms():
    state = inputs()
    out = apply(state, prediction(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 5),
                                  torch.tensor([False, True])))
    assert torch.equal(out.protein_pos_next[:6], state.protein_pos[:6])
    assert torch.equal(out.applied_chi[0], torch.zeros(5))
    assert torch.equal(out.applied_chi[:, 1:], torch.zeros(2, 4))


def test_proline_is_rigid_only():
    state = replace(inputs(), protein_atom_name=['N', 'CA', 'C', 'CB', 'CG', 'CD'] * 2)
    rigid = prediction(tr=torch.ones(2, 3) * .1)
    joint = prediction(tr=rigid.remaining_translation_local, chi=torch.ones(2, 5))
    first = apply(state, rigid, residues=['PRO', 'PRO'])
    second = apply(state, joint, residues=['PRO', 'PRO'])
    assert torch.equal(first.protein_pos_next, second.protein_pos_next)
    assert not second.applied_chi.any()
    assert not torch.equal(second.protein_pos_next, state.protein_pos)


def test_degenerate_current_frame_is_frozen_even_with_nonzero_prediction():
    state = inputs()
    current = state.protein_pos.clone()
    current[0] = current[1]  # N and CA coincide in residue 0 only.
    state = replace(state, protein_pos=current)
    out = apply(state, prediction(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 5)))
    assert not out.diagnostics['frame_valid'][0]
    assert torch.equal(out.protein_pos_next[:6], current[:6])
    assert not torch.equal(out.protein_pos_next[6:], current[6:])


def test_mixed_graph_batch_matches_independent_steps():
    state, network = inputs(), model(nonzero=True)
    batch = predict_joint_step(network, state).protein_pos_next
    for i in range(2):
        atom_slice = slice(i * 6, (i + 1) * 6)
        single = replace(state, protein_pos=state.protein_pos[atom_slice],
                         apo_pos_ref=state.apo_pos_ref[atom_slice], protein_feature=state.protein_feature[atom_slice],
                         atom_to_residue_global=torch.zeros(6, dtype=torch.long),
                         residue_type=state.residue_type[i:i+1], batch_protein=torch.zeros(6, dtype=torch.long),
                         batch_residue=torch.zeros(1, dtype=torch.long), ligand_pos=state.ligand_pos[i:i+1],
                         ligand_v=state.ligand_v[i:i+1], batch_ligand=torch.zeros(1, dtype=torch.long),
                         targetdiff_t=state.targetdiff_t[i:i+1], pocket_k=state.pocket_k[i:i+1],
                         protein_atom_name=state.protein_atom_name[atom_slice])
        torch.testing.assert_close(predict_joint_step(network, single).protein_pos_next,
                                   batch[atom_slice], atol=1e-6, rtol=1e-6)


def test_wrap_chi_before_fractional_step():
    chi = torch.zeros(2, 5)
    chi[:, 0] = 2 * math.pi + .6
    out = apply(inputs(), prediction(chi=chi), steps=2)
    torch.testing.assert_close(out.applied_chi[:, 0], torch.full((2,), .3), atol=1e-6, rtol=0)


def test_full_step_matches_phase31_oracle():
    from pocketdiff.geometry import build_residue_frames, remaining_transform_current_to_holo
    from pocketdiff.geometry.current_state import build_current_chi_state
    state = inputs()
    target = state.protein_pos.clone()
    target[[5, 11], 1] -= 1.
    target[[5, 11], 2] += 1.
    target = target @ so3_exp(torch.tensor([.1, .2, -.1])).T + .2
    args = (state.atom_to_residue_global, state.protein_atom_name)
    cur_f = build_residue_frames(state.protein_pos, *args, num_residues=2)
    tar_f = build_residue_frames(target, *args, num_residues=2)
    remaining = remaining_transform_current_to_holo(cur_f.origins, cur_f.frames, tar_f.origins, tar_f.frames)
    current = build_current_chi_state(state.protein_pos, args[1], args[0], ['CYS'] * 2)
    end = build_current_chi_state(target, args[1], args[0], ['CYS'] * 2)
    delta = periodic_chi_delta(current.angles, end.angles, current.geometry_rotatable_mask & end.geometry_rotatable_mask)
    actual = apply(state, prediction(remaining.translation_local, remaining.rotvec_local, delta))
    oracle = oracle_rigid_chi_reconstruction(state.protein_pos, target, *args, ['CYS'] * 2, remaining.valid)
    torch.testing.assert_close(actual.protein_pos_next, oracle.rigid_chi_positions, atol=3e-6, rtol=0)


@pytest.mark.parametrize('steps', [0, 21, True, 1.5, torch.tensor([1])])
def test_reject_invalid_remaining_steps(steps):
    with pytest.raises((ValueError, TypeError)):
        apply(inputs(), prediction(), steps=steps)
