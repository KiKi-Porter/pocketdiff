import math

import pytest
import torch

from pocketdiff.geometry import extract_chi_angles, periodic_chi_delta


def test_extract_chi_angle_and_mask():
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    angles, mask = extract_chi_angles(
        positions, ['N', 'CA', 'CB', 'SG'], torch.zeros(4, dtype=torch.long), ['CYS'],
    )
    assert bool(mask[0, 0])
    assert angles[0, 0].item() == pytest.approx(-math.pi / 2, abs=1e-6)
    assert not bool(mask[0, 1:].any())


def test_missing_or_degenerate_chi_is_masked():
    positions = torch.zeros(3, 3)
    angles, mask = extract_chi_angles(
        positions, ['N', 'CA', 'CB'], torch.zeros(3, dtype=torch.long), ['CYS'],
    )
    assert torch.equal(angles, torch.zeros(1, 5))
    assert not bool(mask.any())


def test_periodic_chi_delta_wraps_boundary_and_masks():
    apo = torch.tensor([[math.pi - 0.1, 0.0], [0.0, 0.0]])
    holo = torch.tensor([[-math.pi + 0.1, 1.0], [1.0, 0.0]])
    mask = torch.tensor([[True, False], [False, True]])
    delta = periodic_chi_delta(apo, holo, mask)
    assert delta[0, 0].item() == pytest.approx(0.2, abs=1e-6)
    assert delta[0, 1].item() == 0.0
    assert delta[1, 0].item() == 0.0


def test_chi_is_invariant_to_translation_and_rotation():
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = positions @ rotation.T + torch.tensor([4.0, -2.0, 1.5])
    first, first_mask = extract_chi_angles(
        positions, ['N', 'CA', 'CB', 'SG'], torch.zeros(4, dtype=torch.long), ['CYS'],
    )
    second, second_mask = extract_chi_angles(
        transformed, ['N', 'CA', 'CB', 'SG'], torch.zeros(4, dtype=torch.long), ['CYS'],
    )
    assert torch.equal(first_mask, second_mask)
    assert torch.allclose(first, second, atol=1e-6)
