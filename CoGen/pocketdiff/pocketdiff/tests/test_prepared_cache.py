import torch
from typing import List

from pocketdiff.data.schema import PocketComplex
from pocketdiff.training.diffusion import DiffusionTrainConfig, load_prepared_diffusion_slices


def _minimal_complex(sample_id: str, frame_valid: List[bool]) -> PocketComplex:
    num_residues = len(frame_valid)
    atom_to_residue = torch.arange(num_residues, dtype=torch.long)
    positions = torch.arange(num_residues * 3, dtype=torch.float32).reshape(-1, 3)
    return PocketComplex(
        sample_id=sample_id,
        protein_pos_apo=positions,
        protein_pos_holo=positions.clone(),
        protein_feature=torch.zeros(num_residues, 27),
        protein_element=torch.full((num_residues,), 6, dtype=torch.long),
        protein_atom_name=["CA"] * num_residues,
        protein_residue_name=["ALA"] * num_residues,
        atom_to_residue=atom_to_residue,
        residue_type=torch.zeros(num_residues, dtype=torch.long),
        residue_chain_id=["A"] * num_residues,
        residue_sequence_id=[str(index + 1) for index in range(num_residues)],
        frame_valid=torch.tensor(frame_valid, dtype=torch.bool),
        chi_apo=torch.zeros(num_residues, 5),
        chi_holo=torch.zeros(num_residues, 5),
        chi_mask=torch.zeros(num_residues, 5, dtype=torch.bool),
        ligand_pos_ref=torch.zeros(1, 3),
        ligand_type_ref=torch.zeros(1, dtype=torch.long),
        center_offset=torch.zeros(3),
    )


def test_prepared_loader_skips_empty_frame_supervision(tmp_path):
    path = tmp_path / "prepared.pt"
    torch.save(
        {
            "format": "pocketdiff-diffusion-v46-cache-v1",
            "cache_version": "v46-bridge-velocity-static-topology-v1",
            "preprocessing_contract": (
                "protein27-ligand13-kabsch-frame-index-sequence-metadata-no-dynamic-graph"
            ),
            "cache_hash": None,
            "splits": {
                "train": [
                    _minimal_complex("bad", [False]),
                    _minimal_complex("good", [True]),
                ],
                "valid": [_minimal_complex("valid", [True])],
                "test": [_minimal_complex("test", [True])],
            },
        },
        path,
    )
    import pocketdiff.preprocessing.diffusion_cache as cache_module

    original_loader = cache_module.load_diffusion_cache
    cache_module.load_diffusion_cache = lambda _: torch.load(
        path, map_location="cpu", weights_only=False
    )
    try:
        config = DiffusionTrainConfig(
            prepared_cache=str(path),
            train_count=1,
            valid_count=1,
            test_count=1,
            holdout_count=1,
        )
        result = load_prepared_diffusion_slices(path, config)
    finally:
        cache_module.load_diffusion_cache = original_loader
    assert result["train"].sample_ids == ["good"]
