"""Phase 12 tests for official-style initial ligand state generation."""

import pytest
import torch

from pocketdiff.targetdiff import TargetDiffAdapter, restore_center


class _InitFakeTargetDiff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.num_classes = 13
        self.num_timesteps = 4
        self.posterior_logvar = torch.full((4,), -2.0)


def _inputs():
    protein_pos = torch.tensor(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        dtype=torch.float32,
    )
    protein_v = torch.zeros(3, 27)
    batch_protein = torch.tensor([0, 0, 1], dtype=torch.long)
    batch_ligand = torch.tensor([0, 0, 1], dtype=torch.long)
    return protein_pos, protein_v, batch_protein, batch_ligand


def test_initialization_matches_official_random_order_and_centering():
    adapter = TargetDiffAdapter(_InitFakeTargetDiff())
    protein_pos, protein_v, batch_protein, batch_ligand = _inputs()
    generator = torch.Generator(device="cpu").manual_seed(17)
    state = adapter.initialize_sampling_state(
        protein_pos=protein_pos,
        protein_v=protein_v,
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        generator=generator,
    )

    expected_generator = torch.Generator(device="cpu").manual_seed(17)
    center = torch.tensor([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    raw_center = center[batch_ligand]
    position_noise = torch.randn(raw_center.shape, generator=expected_generator)
    uniform = torch.rand((batch_ligand.shape[0], 13), generator=expected_generator)
    gumbel = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    expected_v = gumbel.argmax(dim=-1)
    expected_raw_ligand = raw_center + position_noise
    expected_centered_ligand = expected_raw_ligand - center[batch_ligand]

    assert torch.equal(state.ligand_pos, expected_centered_ligand)
    assert torch.equal(state.ligand_v, expected_v)
    assert torch.equal(state.center_offset, center)
    assert torch.equal(
        restore_center(state.ligand_pos, state.batch_ligand, state.center_offset),
        expected_raw_ligand,
    )
    assert torch.equal(state.protein_pos.mean(dim=0), torch.zeros(3))


def test_initialization_is_generator_replayable_and_does_not_mutate_inputs():
    adapter = TargetDiffAdapter(_InitFakeTargetDiff())
    protein_pos, protein_v, batch_protein, batch_ligand = _inputs()
    before = {
        "protein_pos": protein_pos.clone(),
        "protein_v": protein_v.clone(),
        "batch_protein": batch_protein.clone(),
        "batch_ligand": batch_ligand.clone(),
    }
    first = adapter.initialize_sampling_state(
        protein_pos=protein_pos,
        protein_v=protein_v,
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        generator=torch.Generator(device="cpu").manual_seed(91),
    )
    second = adapter.initialize_sampling_state(
        protein_pos=protein_pos,
        protein_v=protein_v,
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        generator=torch.Generator(device="cpu").manual_seed(91),
    )
    assert torch.equal(first.ligand_pos, second.ligand_pos)
    assert torch.equal(first.ligand_v, second.ligand_v)
    assert all(torch.equal(value, before[name]) for name, value in before.items())


def test_initialization_rejects_nonprotein_center_and_invalid_batches():
    adapter = TargetDiffAdapter(_InitFakeTargetDiff())
    protein_pos, protein_v, batch_protein, batch_ligand = _inputs()
    kwargs = dict(
        protein_pos=protein_pos,
        protein_v=protein_v,
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
    )
    with pytest.raises(ValueError, match="center_mode='protein'"):
        adapter.initialize_sampling_state(**kwargs, center_mode="none")
    with pytest.raises(ValueError, match="out-of-range"):
        out_of_range = dict(kwargs)
        out_of_range["batch_ligand"] = torch.tensor([0, 1, 2], dtype=torch.long)
        adapter.initialize_sampling_state(
            **out_of_range,
        )
    with pytest.raises(ValueError, match="non-empty"):
        empty_ligand = dict(kwargs)
        empty_ligand["batch_ligand"] = torch.empty(0, dtype=torch.long)
        adapter.initialize_sampling_state(
            **empty_ligand,
        )
