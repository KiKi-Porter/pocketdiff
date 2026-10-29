import math

import pytest
import torch

from pocketdiff.geometry import build_current_chi_state, periodic_chi_delta


@pytest.fixture
def cys_coordinates():
    names = ['N', 'CA', 'C', 'O', 'CB', 'SG']
    pos = torch.tensor([[0., 1., 0.], [0., 0., 0.], [.3, -1., 0.],
                        [.5, -2., .4], [1., 0., 0.], [1., 1., 0.]])
    return pos, names, torch.zeros(len(names), dtype=torch.long)


def test_current_angles_follow_current_coordinates(cys_coordinates):
    pos, names, ids = cys_coordinates
    before = pos.clone()
    initial = build_current_chi_state(pos, names, ids, ['CYS'])
    moved = pos.clone()
    moved[-1] = torch.tensor([1., 0., 1.])
    current = build_current_chi_state(moved, names, ids, ['CYS'])
    assert initial.geometry_rotatable_mask.tolist() == [[True, False, False, False, False]]
    delta = periodic_chi_delta(initial.angles, current.angles, current.geometry_rotatable_mask)
    assert float(delta[0, 0]) == pytest.approx(math.pi / 2, abs=1e-6)
    assert torch.equal(pos, before)


def test_geometry_mask_does_not_substitute_terminal_alias():
    # ASP OD2 can be read by the historical extractor, but is not a canonical
    # substitute for a missing OD1 in the new coordinate update contract.
    names = ['N', 'CA', 'C', 'CB', 'CG', 'OD2']
    pos = torch.randn(len(names), 3, generator=torch.Generator().manual_seed(31))
    state = build_current_chi_state(pos, names, torch.zeros(len(names), dtype=torch.long), ['ASP'])
    assert state.geometry_rotatable_mask[0, 0]
    assert not state.geometry_rotatable_mask[0, 1]
    assert state.angles[0, 1] == 0


@pytest.mark.parametrize('residue,names', [
    ('ALA', ['N', 'CA', 'C', 'CB']), ('GLY', ['N', 'CA', 'C']),
    ('PRO', ['N', 'CA', 'C', 'CB', 'CG', 'CD']),
])
def test_nonrotatable_residues_are_explicitly_masked(residue, names):
    pos = torch.randn(len(names), 3, generator=torch.Generator().manual_seed(31))
    state = build_current_chi_state(pos, names, torch.zeros(len(names), dtype=torch.long), [residue])
    assert not state.geometry_rotatable_mask.any()
    assert torch.equal(state.angles, torch.zeros_like(state.angles))


def test_degenerate_current_quartet_is_not_rotatable(cys_coordinates):
    pos, names, ids = cys_coordinates
    pos[-1] = torch.tensor([2., 0., 0.])  # SG on the CA-CB axis
    state = build_current_chi_state(pos, names, ids, ['CYS'])
    assert not state.geometry_rotatable_mask.any()


def test_nonfinite_current_is_rejected_instead_of_silently_masked(cys_coordinates):
    pos, names, ids = cys_coordinates
    pos[-1, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        build_current_chi_state(pos, names, ids, ['CYS'])


def test_ambiguity_annotation_does_not_disable_geometry():
    names = ['N', 'CA', 'C', 'CB', 'CG', 'OD1', 'OD2']
    pos = torch.randn(len(names), 3, generator=torch.Generator().manual_seed(31))
    state = build_current_chi_state(pos, names, torch.zeros(len(names), dtype=torch.long), ['ASP'])
    assert state.geometry_rotatable_mask[0, :2].all()
    assert state.ambiguous_chi_mask.tolist() == [[False, True, False, False, False]]
