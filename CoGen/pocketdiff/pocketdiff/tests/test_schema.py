import pytest
import torch

from pocketdiff.data.schema import PocketBatchState, PocketComplex


def _complex() -> PocketComplex:
    return PocketComplex(
        sample_id="toy",
        protein_pos_apo=torch.zeros(4, 3),
        protein_pos_holo=torch.ones(4, 3),
        protein_feature=torch.zeros(4, 27),
        protein_element=torch.tensor([7, 6, 6, 7]),
        protein_atom_name=["N", "CA", "C", "N"],
        protein_residue_name=["ALA"] * 4,
        atom_to_residue=torch.tensor([0, 0, 0, 1]),
        residue_type=torch.tensor([0, 1]),
        residue_chain_id=["A", "A"],
        residue_sequence_id=["1", "2"],
        frame_valid=torch.tensor([True, False]),
        chi_apo=torch.zeros(2, 5),
        chi_holo=torch.zeros(2, 5),
        chi_mask=torch.zeros(2, 5, dtype=torch.bool),
        ligand_pos_ref=torch.zeros(2, 3),
        ligand_type_ref=torch.tensor([0, 12]),
        center_offset=torch.zeros(3),
    )


def test_pocket_complex_accepts_the_frozen_contract():
    value = _complex()
    assert value.num_protein_atoms == 4
    assert value.num_residues == 2
    assert value.num_ligand_atoms == 2


def test_pocket_complex_rejects_wrong_protein_feature_width():
    value = _complex()
    value.protein_feature = torch.zeros(4, 28)
    with pytest.raises(ValueError, match="protein_feature"):
        value.__post_init__()


def test_pocket_complex_rejects_integer_coordinates():
    value = _complex()
    value.protein_pos_apo = torch.zeros(4, 3, dtype=torch.long)
    with pytest.raises(TypeError, match="floating dtype"):
        value.__post_init__()


def test_batch_state_rejects_cross_graph_residue_mapping():
    with pytest.raises(ValueError, match="crosses graph boundaries"):
        PocketBatchState(
            protein_pos=torch.zeros(4, 3),
            apo_pos_ref=torch.zeros(4, 3),
            protein_feature=torch.zeros(4, 27),
            atom_to_residue_local=torch.tensor([0, 0, 0, 0]),
            atom_to_residue_global=torch.tensor([0, 1, 0, 1]),
            batch_protein=torch.tensor([0, 0, 1, 1]),
            batch_residue=torch.tensor([0, 1]),
            ligand_pos=torch.zeros(0, 3),
            ligand_v=torch.zeros(0, dtype=torch.long),
            batch_ligand=torch.zeros(0, dtype=torch.long),
            k=torch.tensor([0, 0]),
            targetdiff_t=torch.tensor([199, 199]),
            pocket_s=torch.tensor([0.0, 0.0]),
            center_offset=torch.zeros(2, 3),
        )
