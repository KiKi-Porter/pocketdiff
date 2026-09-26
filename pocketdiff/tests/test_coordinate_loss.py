import pytest
import torch
from pocketdiff.training import masked_next_xyz_loss


def test_next_xyz_loss_is_graph_balanced_and_apo_frame_based():
    target = torch.zeros(5, 3)
    predicted = target.clone()
    predicted[:4, 0] = 1.
    predicted[4, 0] = 3.
    atom_to_residue = torch.tensor([0, 0, 0, 1, 2])
    frame_valid = torch.tensor([True, True, True])
    batch = torch.tensor([0, 0, 0, 0, 1])
    # per-coordinate MSE: graph0=1/3, graph1=3; graph-balanced=5/3
    out = masked_next_xyz_loss(predicted, target, atom_to_residue, frame_valid, batch)
    assert out.valid_atom_count == 5 and out.valid_graph_count == 2
    assert out.loss.item() == pytest.approx(5./3.)
    assert out.graph_losses.tolist() == pytest.approx([1./3., 3.])


def test_next_xyz_loss_excludes_invalid_residue_and_empty_graph():
    target = torch.zeros(4, 3)
    predicted = torch.ones(4, 3)
    ids = torch.tensor([0, 0, 1, 1])
    valid = torch.tensor([True, False])
    batch = torch.tensor([0, 0, 1, 1])
    out = masked_next_xyz_loss(predicted, target, ids, valid, batch)
    assert out.valid_atom_count == 2 and out.valid_graph_count == 1
    assert out.loss.item() == pytest.approx(1.)


def test_next_xyz_loss_backpropagates_and_rejects_empty_mask():
    predicted = torch.zeros(2, 3, requires_grad=True)
    target = torch.ones(2, 3)
    out = masked_next_xyz_loss(predicted, target, torch.zeros(2, dtype=torch.long),
                               torch.ones(1, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
    out.loss.backward()
    assert torch.isfinite(predicted.grad).all()
    with pytest.raises(ValueError, match='no valid graph'):
        masked_next_xyz_loss(predicted.detach(), target, torch.zeros(2, dtype=torch.long),
                             torch.zeros(1, dtype=torch.bool), torch.zeros(2, dtype=torch.long))


def test_next_xyz_loss_does_not_realign_or_accept_bad_mapping():
    with pytest.raises(ValueError, match='invalid residue'):
        masked_next_xyz_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([0, 2]),
                             torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
