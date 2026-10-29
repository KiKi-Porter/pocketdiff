"""Phase 14 tests for read-only apo-to-holo geometry diagnostics."""

import pytest
import torch

from pocketdiff.evaluation import evaluate_protein_motion


def test_identity_geometry_has_zero_rmsd_and_displacement():
    apo = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    offset = torch.tensor([[1.0, 0.0, 0.0]])
    batch = torch.zeros(2, dtype=torch.long)
    diagnostics = evaluate_protein_motion(apo, apo, apo - offset, offset, batch)
    assert diagnostics.apo_to_holo_rmsd == 0.0
    assert diagnostics.final_to_holo_rmsd == 0.0
    assert diagnostics.rmsd_improvement == 0.0
    assert diagnostics.max_final_displacement == 0.0
    assert diagnostics.mean_final_displacement == 0.0


def test_known_translation_and_perfect_final_motion_are_reported():
    apo = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    holo = apo + torch.tensor([0.0, 1.0, 0.0])
    offset = torch.tensor([[1.0, 0.0, 0.0]])
    batch = torch.zeros(2, dtype=torch.long)
    diagnostics = evaluate_protein_motion(apo, holo, holo - offset, offset, batch)
    assert diagnostics.apo_to_holo_rmsd == pytest.approx(1.0)
    assert diagnostics.final_to_holo_rmsd == pytest.approx(0.0)
    assert diagnostics.rmsd_improvement == pytest.approx(1.0)
    assert diagnostics.max_final_displacement == pytest.approx(1.0)
    assert diagnostics.mean_final_displacement == pytest.approx(1.0)


def test_batched_offsets_are_applied_per_atom():
    apo = torch.tensor([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    holo = apo + torch.tensor([0.0, 0.5, 0.0])
    offset = torch.tensor([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    batch = torch.tensor([0, 1], dtype=torch.long)
    diagnostics = evaluate_protein_motion(apo, holo, torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]), offset, batch)
    assert diagnostics.apo_to_holo_rmsd == pytest.approx(0.5)
    assert diagnostics.final_to_holo_rmsd == pytest.approx(0.0)


def test_geometry_diagnostics_rejects_shape_and_finite_errors():
    apo = torch.zeros(2, 3)
    batch = torch.zeros(2, dtype=torch.long)
    offset = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="shape"):
        evaluate_protein_motion(apo, torch.zeros(3, 3), apo, offset, batch)
    with pytest.raises(ValueError, match="finite"):
        bad = apo.clone()
        bad[0, 0] = float("nan")
        evaluate_protein_motion(apo, bad, apo, offset, batch)
