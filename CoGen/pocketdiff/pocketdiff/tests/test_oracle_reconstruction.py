import math

import pytest
import torch

from pocketdiff.geometry import oracle_rigid_chi_reconstruction
from pocketdiff.tests.test_current_chi import cys_coordinates


@pytest.mark.parametrize('start,finish', [(0., math.pi / 2), (3.1, -3.1), (-1., 2.)])
def test_oracle_recovers_independently_constructed_rigid_plus_chi(cys_coordinates, start, finish):
    apo, names, ids = cys_coordinates
    apo[-1] = torch.tensor([1., math.cos(start), math.sin(start)])
    target = apo.clone()
    target[-1] = torch.tensor([1., math.cos(finish), math.sin(finish)])
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    holo = target @ rotation.T + torch.tensor([1.3, -.2, 2.])
    before = apo.clone(), holo.clone()
    out = oracle_rigid_chi_reconstruction(apo, holo, ids, names, ['CYS'], torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(out.rigid_chi_positions, holo, atol=2e-6, rtol=0)
    assert out.rigid_chi_metrics.atom_rmsd < 2e-6
    assert out.rigid_metrics.atom_rmsd > .01
    assert out.rigid_metrics.backbone_rmsd < 2e-6
    assert torch.equal(out.rigid_only_positions[:4], out.rigid_chi_positions[:4])
    assert torch.equal(apo, before[0]) and torch.equal(holo, before[1])


def test_holo_degeneracy_changes_supervision_but_never_current_state(cys_coordinates):
    apo, names, ids = cys_coordinates
    holo = apo.clone()
    holo[-1] = torch.tensor([2., 0., 0.])
    out = oracle_rigid_chi_reconstruction(apo, holo, ids, names, ['CYS'], torch.ones(1, dtype=torch.bool))
    baseline = oracle_rigid_chi_reconstruction(apo, apo, ids, names, ['CYS'], torch.ones(1, dtype=torch.bool))
    assert out.current_chi.geometry_rotatable_mask[0, 0]
    assert not out.supervision_mask.any()
    assert baseline.supervision_mask[0, 0]
    assert torch.equal(out.current_chi.angles, baseline.current_chi.angles)
    assert torch.equal(out.current_chi.geometry_rotatable_mask, baseline.current_chi.geometry_rotatable_mask)
    assert torch.equal(out.rigid_chi_positions, out.rigid_only_positions)


def test_invalid_frame_freezes_residue_but_not_current_chi_observation(cys_coordinates):
    apo, names, ids = cys_coordinates
    out = oracle_rigid_chi_reconstruction(apo, apo + 3., ids, names, ['CYS'], torch.zeros(1, dtype=torch.bool))
    assert torch.equal(out.rigid_chi_positions, apo)
    assert not out.supervision_mask.any()
    assert out.current_chi.geometry_rotatable_mask[0, 0]
    assert out.rigid_chi_metrics.valid_atom_count == 0


def test_ambiguous_slot_kept_for_named_oracle_but_excluded_from_training():
    names = ['N', 'CA', 'C', 'CB', 'CG', 'OD1', 'OD2']
    pos = torch.randn(len(names), 3, generator=torch.Generator().manual_seed(31))
    out = oracle_rigid_chi_reconstruction(pos, pos, torch.zeros(len(names), dtype=torch.long),
                                        names, ['ASP'], torch.ones(1, dtype=torch.bool))
    assert out.supervision_mask[0, :2].all()
    assert out.training_safe_mask[0, 0]
    assert not out.training_safe_mask[0, 1]


def test_oracle_rejects_mismatched_coordinates(cys_coordinates):
    apo, names, ids = cys_coordinates
    with pytest.raises(ValueError, match='shape'):
        oracle_rigid_chi_reconstruction(apo, apo[:-1], ids, names, ['CYS'], torch.ones(1, dtype=torch.bool))
