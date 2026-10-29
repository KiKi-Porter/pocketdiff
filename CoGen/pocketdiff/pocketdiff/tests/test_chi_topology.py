import math

import torch

from pocketdiff.geometry import apply_chi_updates, build_chi_update_metadata


def _atom_indices(mask):
    return torch.where(mask)[0].tolist()


def test_chi_topology_builds_linear_and_branched_sidechains():
    names = [
        "N", "CA", "C", "CB", "SG",  # CYS residue 0
        "N", "CA", "C", "CB", "CG1", "CG2",  # VAL residue 1
        "N", "CA", "C", "CB", "CG", "CD", "CE", "NZ",  # LYS residue 2
    ]
    atom_to_residue = torch.tensor([0] * 5 + [1] * 6 + [2] * 8, dtype=torch.long)
    metadata = build_chi_update_metadata(
        names, atom_to_residue, ["CYS", "VAL", "LYS"]
    )
    assert metadata.axis_start.tolist()[0][0] == 1
    assert metadata.axis_end.tolist()[0][0] == 3
    assert _atom_indices(metadata.downstream_atom_mask[0, 0]) == [4]
    # VAL χ1 rotates both methyl branches from the CB axis.
    assert _atom_indices(metadata.downstream_atom_mask[1, 0]) == [9, 10]
    # LYS χ1..χ4 successively shorten the distal chain.
    assert [_atom_indices(metadata.downstream_atom_mask[2, chi]) for chi in range(4)] == [
        [15, 16, 17, 18], [16, 17, 18], [17, 18], [18]
    ]
    assert bool(metadata.valid[0, 0]) and bool(metadata.valid[1, 0])
    assert bool(metadata.valid[2, :4].all())
    assert not bool(metadata.valid[:, 4].any())


def test_missing_terminal_atom_is_invalid_and_not_rotated():
    names = ["N", "CA", "C", "CB"]
    metadata = build_chi_update_metadata(
        names, torch.zeros(4, dtype=torch.long), ["CYS"]
    )
    assert not bool(metadata.valid.any())
    assert torch.equal(metadata.axis_start, torch.full((1, 5), -1, dtype=torch.long))
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [1., 0., 1.]])
    result = apply_chi_updates(
        positions,
        metadata.axis_start,
        metadata.axis_end,
        metadata.downstream_atom_mask,
        torch.full((1, 5), math.pi / 2),
        valid=metadata.valid,
    )
    assert torch.equal(result.positions, positions)
    assert not bool(result.valid.any())


def test_built_metadata_drives_expected_cys_rotation():
    names = ["N", "CA", "C", "CB", "SG"]
    positions = torch.tensor([
        [0., 1., 0.], [0., 0., 0.], [2., 0., 0.], [1., 0., 0.], [1., 1., 0.]
    ])
    metadata = build_chi_update_metadata(
        names, torch.zeros(5, dtype=torch.long), ["CYS"]
    )
    delta = torch.zeros(1, 5)
    delta[0, 0] = math.pi / 2
    result = apply_chi_updates(
        positions,
        metadata.axis_start,
        metadata.axis_end,
        metadata.downstream_atom_mask,
        delta,
        valid=metadata.valid,
    )
    torch.testing.assert_close(result.positions[4], torch.tensor([1., 0., 1.]), atol=1e-6, rtol=0)
    assert bool(result.valid[0, 0])


def test_complete_rings_and_terminal_branches_move_together():
    expected = {
        'PHE': ['CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
        'TYR': ['CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
        'HIS': ['ND1', 'CD2', 'CE1', 'NE2'],
        'TRP': ['CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
        'ARG': ['CD', 'NE', 'CZ', 'NH1', 'NH2'],
    }
    for residue, distal in expected.items():
        names = ['N', 'CA', 'C', 'O', 'CB', 'CG'] + distal
        meta = build_chi_update_metadata(names, torch.zeros(len(names), dtype=torch.long), [residue])
        assert {names[i] for i in _atom_indices(meta.downstream_atom_mask[0, 1])} == set(distal)
    names = ['N', 'CA', 'C', 'CB', 'OG1', 'CG2']
    meta = build_chi_update_metadata(names, torch.zeros(6, dtype=torch.long), ['THR'])
    assert _atom_indices(meta.downstream_atom_mask[0, 0]) == [4, 5]


def test_ile_second_rotation_never_moves_other_branch_or_substitutes_axis():
    names = ['N', 'CA', 'C', 'CB', 'CG1', 'CG2', 'CD1']
    meta = build_chi_update_metadata(names, torch.zeros(7, dtype=torch.long), ['ILE'])
    assert _atom_indices(meta.downstream_atom_mask[0, 1]) == [6]
    names.remove('CG1')
    meta = build_chi_update_metadata(names, torch.zeros(6, dtype=torch.long), ['ILE'])
    assert not meta.valid.any()


def test_proline_ring_and_backbone_remain_frozen():
    names = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD']
    meta = build_chi_update_metadata(names, torch.zeros(7, dtype=torch.long), ['PRO'])
    assert not meta.valid.any()
    assert not meta.downstream_atom_mask.any()


def test_truncating_output_slots_keeps_complete_distal_atoms():
    names = ['N', 'CA', 'C', 'CB', 'CG', 'CD', 'CE', 'NZ']
    meta = build_chi_update_metadata(names, torch.zeros(8, dtype=torch.long), ['LYS'], num_chi=1)
    assert _atom_indices(meta.downstream_atom_mask[0, 0]) == [4, 5, 6, 7]
    assert not meta.valid[0, 1:].any()


def test_unknown_and_duplicate_atoms_are_explicit_errors():
    import pytest
    for names in (['N', 'CA', 'CB', 'SG', 'HB1'], ['N', 'CA', 'CB', 'SG', 'SG']):
        with pytest.raises(ValueError):
            build_chi_update_metadata(names, torch.zeros(len(names), dtype=torch.long), ['CYS'])


def test_all_residue_rotations_preserve_bonds_and_read_back_requested_angles():
    from pocketdiff.geometry.chi import SIDECHAIN_BONDS, extract_chi_angles, periodic_chi_delta
    generator = torch.Generator().manual_seed(30)
    for residue, bonds in SIDECHAIN_BONDS.items():
        edges = [pair.split('-') for pair in ('N-CA CA-C C-O ' + bonds).split()]
        names = sorted({atom for edge in edges for atom in edge})
        ids = torch.zeros(len(names), dtype=torch.long)
        # Nondegenerate coordinates test the kinematics independently of any
        # ideal residue geometry or the implementation's own reference frames.
        pos = torch.randn(len(names), 3, generator=generator, dtype=torch.float64)
        before = pos.clone()
        meta = build_chi_update_metadata(names, ids, [residue])
        angles, mask = extract_chi_angles(pos, names, ids, [residue])
        delta = torch.tensor([[.31, -.27, .19, -.42, 0.]], dtype=pos.dtype)
        result = apply_chi_updates(pos, meta.axis_start, meta.axis_end,
                                   meta.downstream_atom_mask, delta, valid=meta.valid)
        updated, updated_mask = extract_chi_angles(result.positions, names, ids, [residue])
        active = meta.valid & mask & updated_mask
        torch.testing.assert_close(periodic_chi_delta(angles, updated, active)[active],
                                   delta[active], atol=1e-10, rtol=0)
        edge_indices = torch.tensor([[names.index(a), names.index(b)] for a, b in edges])
        def lengths(coords):
            return (coords[edge_indices[:, 0]] - coords[edge_indices[:, 1]]).norm(dim=-1)
        torch.testing.assert_close(lengths(pos), lengths(result.positions), atol=1e-10, rtol=0)
        assert torch.equal(pos, before)
        for atom in ('N', 'CA', 'C', 'O'):
            assert torch.equal(pos[names.index(atom)], result.positions[names.index(atom)])


def test_zero_rotation_retains_exact_coordinates_and_useful_gradient():
    pos = torch.tensor([[.21, .3, -.19], [1.2, .7, .18], [1.4, 1.9, -.41]], dtype=torch.float64)
    start = torch.tensor([[0, -1, -1, -1, -1]])
    end = torch.tensor([[1, -1, -1, -1, -1]])
    mask = torch.zeros(1, 5, 3, dtype=torch.bool)
    mask[0, 0, 2] = True
    delta = torch.zeros(1, 5, dtype=pos.dtype, requires_grad=True)
    result = apply_chi_updates(pos, start, end, mask, delta)
    assert torch.equal(result.positions, pos)
    result.positions[2, 2].backward()
    assert delta.grad[0, 0].abs() > .1
    assert torch.isfinite(delta.grad).all()


def test_two_residues_are_isolated_and_updates_are_se3_equivariant():
    from pocketdiff.geometry.so3 import so3_exp
    local = torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., -1., 0.], [1., 0., 0.], [1., 1., .5]])
    pos = torch.cat([local, local + torch.tensor([5., 2., 1.])])
    names = ['N', 'CA', 'C', 'CB', 'SG'] * 2
    meta = build_chi_update_metadata(names, torch.tensor([0] * 5 + [1] * 5), ['CYS'] * 2)
    assert not meta.downstream_atom_mask[0, :, 5:].any()
    assert not meta.downstream_atom_mask[1, :, :5].any()
    delta = torch.tensor([[.5, 0., 0., 0., 0.], [-.3, 0., 0., 0., 0.]])
    def update(p):
        return apply_chi_updates(p, meta.axis_start, meta.axis_end,
                                 meta.downstream_atom_mask, delta, valid=meta.valid).positions
    rotation = so3_exp(torch.tensor([.2, -.3, .4]))
    shift = torch.tensor([3., -2., 4.])
    torch.testing.assert_close(update(pos @ rotation.T + shift), update(pos) @ rotation.T + shift,
                               atol=2e-6, rtol=0)
