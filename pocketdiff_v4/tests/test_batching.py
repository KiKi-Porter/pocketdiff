import torch

from pocketdiff_v4.batching import collate_complexes


def _record(sample_id, atom_offset=0.0):
    apo = torch.tensor(
        [[-0.4, 0.2, 0.1], [0.0, 0.0, 0.0], [0.2, 0.8, -0.1],
         [0.3, -0.5, 0.4], [0.8, -0.2, 0.7]],
        dtype=torch.float32,
    ) + atom_offset
    return {
        "sample_id": sample_id,
        "input": {
            "apo_pos": apo,
            "protein_feature": torch.zeros((5, 27)),
            "ligand_pos": torch.tensor([[2.0, 0.4, -0.2]]) + atom_offset,
            "ligand_type": torch.tensor([1]),
            "atom_to_residue": torch.zeros(5, dtype=torch.long),
            "residue_type": torch.tensor([5]),
            "residue_feature": torch.zeros((1, 27)),
            "residue_center_apo": apo[1:2],
            "frame_index": torch.tensor([[0, 1, 2]]),
            "chi_geometry_mask": torch.zeros((1, 5), dtype=torch.bool),
            "chi_axis": torch.full((1, 5, 2), -1, dtype=torch.long),
            "chi_ptr": torch.zeros(6, dtype=torch.long),
            "chi_downstream": torch.empty(0, dtype=torch.long),
            "rr_edge_index": torch.empty((2, 0), dtype=torch.long),
            "lr_edge_index": torch.tensor([[0], [0]]),
            "ll_edge_index": torch.empty((2, 0), dtype=torch.long),
        },
        "target": {
            "holo_pos": apo.clone(),
            "chi_apo": torch.zeros((1, 5)),
            "chi_holo": torch.zeros((1, 5)),
            "chi_supervision_mask": torch.zeros((1, 5), dtype=torch.bool),
        },
    }


def test_collator_returns_training_contract_and_offsets_indices():
    batch = collate_complexes([_record("a"), _record("b", 4.0)])
    inputs, targets = batch["input"], batch["target"]
    assert inputs["apo_pos"].shape == (10, 3)
    assert inputs["frame_index"].tolist() == [[0, 1, 2], [5, 6, 7]]
    assert inputs["atom_to_residue"].tolist() == [0] * 5 + [1] * 5
    assert inputs["lr_edge_index"].tolist() == [[0, 1], [0, 1]]
    assert inputs["ll_edge_index"].shape == (2, 0)
    assert inputs["chi_ptr"].shape == (11,)
    assert inputs["chi_ptr"][-1].item() == inputs["chi_downstream"].numel()
    assert inputs["atom_ptr"].tolist() == [0, 5, 10]
    assert targets["holo_pos"].shape == (10, 3)
    assert targets["sample_id"] == ["a", "b"]
