from dataclasses import replace
import pytest
import torch

from pocketdiff.models import TargetDiffEncoderAdapter
from pocketdiff.tests.test_current_joint_model import inputs
from pocketdiff.geometry import so3_exp


def batch_inputs():
    state = inputs()
    return (state.protein_pos, state.protein_feature, state.batch_protein,
            state.ligand_pos, state.ligand_v, state.batch_ligand)


def test_targetdiff_encoder_contract_forward_fix_x_and_shapes():
    model = TargetDiffEncoderAdapter()
    out = model(*batch_inputs())
    assert out['protein_hidden'].shape == (12, 128)
    assert out['ligand_hidden'].shape == (2, 128)
    assert out['coordinates'].shape == (14, 3)
    assert torch.isfinite(out['protein_hidden']).all()
    assert torch.equal(out['coordinates'], torch.cat((batch_inputs()[0], batch_inputs()[3])))
    assert model.contract == 'targetdiff-uni-o2-readonly-v1'
    assert model.config == dict(hidden_dim=128, protein_feature_dim=27, ligand_classes=13,
                                num_blocks=1, num_layers=9, heads=16, knn=32, num_rbf=20, r_max=10.0)


def test_targetdiff_encoder_is_se3_invariant_in_hidden_space():
    model = TargetDiffEncoderAdapter()
    pos, feat, bp, lig, lv, bl = batch_inputs()
    rot = so3_exp(torch.tensor([.2, -.3, .4]))
    shift = torch.tensor([2., -.5, .7])
    first = model(pos, feat, bp, lig, lv, bl)
    second = model(pos @ rot.T + shift, feat, bp, lig @ rot.T + shift, lv, bl)
    torch.testing.assert_close(first['protein_hidden'], second['protein_hidden'], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first['ligand_hidden'], second['ligand_hidden'], atol=3e-5, rtol=3e-5)


def test_targetdiff_encoder_supports_mixed_graph_k_neighbors():
    model = TargetDiffEncoderAdapter()
    out = model(*batch_inputs())
    assert out['protein_hidden'].shape[0] == 12


@pytest.mark.parametrize('kwargs', [dict(hidden_dim=64), dict(num_layers=1), dict(heads=8), dict(num_rbf=16)])
def test_targetdiff_encoder_contract_is_frozen(kwargs):
    with pytest.raises(ValueError, match='frozen'):
        TargetDiffEncoderAdapter(**kwargs)


def test_targetdiff_encoder_rejects_bad_features():
    values = list(batch_inputs())
    values[1] = values[1][:, :26]
    with pytest.raises(ValueError, match='protein_feature'):
        TargetDiffEncoderAdapter()(*values)
