import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset

from datasets.bridge_finetune import TargetDiffInputDataset
from datasets.pl_data import ProteinLigandData


def _sample(extra_fields):
    sample = ProteinLigandData(
        protein_element=torch.tensor([6, 7]),
        protein_pos=torch.zeros((2, 3)),
        protein_atom_feature=torch.zeros((2, 27)),
        ligand_element=torch.tensor([6]),
        ligand_pos=torch.ones((1, 3)),
        ligand_atom_feature_full=torch.tensor([1]),
    )
    for name, value in extra_fields.items():
        setattr(sample, name, value)
    return sample


def test_mixed_bridge_and_md_records_collate_to_loss_inputs():
    bridge = _sample({"num_nodes": 3, "protein_atom_name": ["CA", "N"]})
    md = _sample({"ligand_bond_index": torch.zeros((2, 0), dtype=torch.long)})
    dataset = ConcatDataset((
        TargetDiffInputDataset([bridge]),
        TargetDiffInputDataset([md]),
    ))

    batch = next(iter(DataLoader(
        dataset,
        batch_size=2,
        follow_batch=["protein_element", "ligand_element"],
    )))

    assert batch.protein_atom_feature.shape == (4, 27)
    assert batch.ligand_atom_feature_full.shape == (2,)
    assert batch.protein_element_batch.tolist() == [0, 0, 1, 1]
    assert batch.ligand_element_batch.tolist() == [0, 1]
    assert set(batch.keys) == {
        "protein_element",
        "protein_pos",
        "protein_atom_feature",
        "ligand_element",
        "ligand_pos",
        "ligand_atom_feature_full",
        "protein_element_batch",
        "protein_element_ptr",
        "ligand_element_batch",
        "ligand_element_ptr",
    }
