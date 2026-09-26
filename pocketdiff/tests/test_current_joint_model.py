from dataclasses import replace

import pytest
import torch

from pocketdiff.inference import PocketInputs, predict_joint_step
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.models import PocketDiffModel


def inputs():
    local = torch.tensor([[0., 1., 0.], [0., 0., 0.], [.3, -1., 0.],
                          [.5, -2., .4], [1., 0., 0.], [1., 1., 0.]])
    pos = torch.cat((local, local + torch.tensor([5., 2., 1.])))
    ids = torch.tensor([0] * 6 + [1] * 6)
    feature = torch.zeros(12, 27)
    feature[:, 1] = 1.
    feature[:, 7] = 1.
    return PocketInputs(
        protein_pos=pos, apo_pos_ref=pos.clone(), protein_feature=feature,
        atom_to_residue_global=ids, residue_type=torch.tensor([1, 1]),
        batch_protein=ids.clone(), batch_residue=torch.tensor([0, 1]),
        ligand_pos=torch.tensor([[2., 2., .5], [7., 4., 1.5]]), ligand_v=torch.tensor([1, 1]),
        batch_ligand=torch.tensor([0, 1]), targetdiff_t=torch.tensor([199, 9]),
        pocket_k=torch.tensor([0, 19]), protein_atom_name=['N', 'CA', 'C', 'O', 'CB', 'SG'] * 2,
    )


def model(nonzero=False):
    value = PocketDiffModel(encoder_layers=1, knn=4, dropout=0.,
                           predict_chi=True, chi_input_mode='current')
    if nonzero:
        with torch.no_grad():
            generator = torch.Generator().manual_seed(320)
            for head in (value.motion_head, value.current_chi_head):
                head.network[-1].weight.copy_(torch.randn(head.network[-1].weight.shape,
                                                          generator=generator) * .003)
            value.motion_head.network[-1].bias.copy_(torch.tensor([.1, -.05, .08, .03, -.02, .04]))
            value.current_chi_head.network[-1].bias.fill_(.15)
    return value.eval()


def test_current_head_consumes_updated_chi_instead_of_apo():
    state = inputs()
    moved = state.protein_pos.clone()
    moved[5] = torch.tensor([1., 0., 1.])
    state = replace(state, protein_pos=moved)
    network = model()
    captured = []
    handle = network.current_chi_head.register_forward_pre_hook(lambda module, args: captured.append(args))
    network(**state.model_kwargs())
    handle.remove()
    expected = build_current_chi_state(moved, state.protein_atom_name,
                                       state.atom_to_residue_global, ['CYS', 'CYS'])
    apo = build_current_chi_state(state.apo_pos_ref, state.protein_atom_name,
                                  state.atom_to_residue_global, ['CYS', 'CYS'])
    assert torch.equal(captured[0][1], expected.angles)
    assert torch.equal(captured[0][2], expected.geometry_rotatable_mask)
    assert not torch.equal(captured[0][1], apo.angles)


@pytest.mark.parametrize('key', ['chi_apo', 'chi_mask'])
def test_current_model_rejects_legacy_label_inputs(key):
    kwargs = inputs().model_kwargs()
    kwargs[key] = torch.zeros(2, 5)
    with pytest.raises(ValueError, match='rejects legacy'):
        model()(**kwargs)


@pytest.mark.parametrize('config', [dict(predict_chi=False)])
def test_current_mode_requires_chi_head(config):
    with pytest.raises(ValueError, match='requires'):
        PocketDiffModel(chi_input_mode='current', **config)


def test_current_joint_bridge_rate_scales_all_three_heads_by_progress():
    state = inputs()
    remaining = model(nonzero=True)
    bridge = PocketDiffModel(encoder_layers=1, knn=4, dropout=0., predict_chi=True,
                             chi_input_mode='current', motion_parameterization='bridge_rate')
    bridge.load_state_dict(remaining.state_dict(), strict=True)
    # At k=0 the public outputs are unchanged; at k=19 all three heads are
    # reduced by the same (20-k)/20 factor before the joint solver divides.
    k0 = replace(state, pocket_k=torch.zeros(2, dtype=torch.long),
                 targetdiff_t=torch.full((2,), 199, dtype=torch.long))
    k19 = replace(state, pocket_k=torch.full((2,), 19, dtype=torch.long),
                  targetdiff_t=torch.full((2,), 9, dtype=torch.long))
    p0 = remaining(**k0.model_kwargs())
    q0 = bridge(**k0.model_kwargs())
    torch.testing.assert_close(p0.remaining_translation_local, q0.remaining_translation_local)
    torch.testing.assert_close(p0.remaining_rotvec_local, q0.remaining_rotvec_local)
    torch.testing.assert_close(p0.remaining_chi, q0.remaining_chi)
    p19 = remaining(**k19.model_kwargs())
    q19 = bridge(**k19.model_kwargs())
    for actual, expected in ((q19.remaining_translation_local, p19.remaining_translation_local * .05),
                             (q19.remaining_rotvec_local, p19.remaining_rotvec_local * .05),
                             (q19.remaining_chi, p19.remaining_chi * .05)):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_checkpoint_names_guard_chi_semantics_and_new_weights_reload():
    network = model(nonzero=True)
    restored = model()
    restored.load_state_dict(network.state_dict(), strict=True)
    assert network.input_contract == 'pocketdiff-current-joint-v1:encoder=scalar'
    assert torch.equal(predict_joint_step(network, inputs()).protein_pos_next,
                       predict_joint_step(restored, inputs()).protein_pos_next)
    legacy = PocketDiffModel(encoder_layers=1, knn=4, dropout=0., predict_chi=True)
    with pytest.raises(RuntimeError, match='current_chi_head'):
        network.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match='current_chi_head'):
        legacy.load_state_dict(network.state_dict(), strict=True)


def test_inference_inputs_exclude_labels_and_reject_cross_graph_mapping():
    state = inputs()
    assert not {'chi_apo', 'chi_holo', 'chi_mask', 'protein_pos_holo'} & state.model_kwargs().keys()
    bad_ids = state.atom_to_residue_global.clone()
    bad_ids[0] = 1
    with pytest.raises(ValueError, match='crosses graph'):
        replace(state, atom_to_residue_global=bad_ids).model_kwargs()


def test_joint_entry_rejects_legacy_model():
    with pytest.raises(ValueError, match='current-chi'):
        predict_joint_step(PocketDiffModel(), inputs())
