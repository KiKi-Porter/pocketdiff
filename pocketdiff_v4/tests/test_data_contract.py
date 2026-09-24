import torch

from pocketdiff.geometry.chi import build_chi_update_metadata
from pocketdiff_v4.constants import TARGETDIFF_RESIDUE_NAMES


def test_targetdiff_residue_order_matches_official_adapter_order():
    assert TARGETDIFF_RESIDUE_NAMES == (
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
        "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
        "THR", "VAL", "TRP", "TYR",
    )


def test_chi_alias_uses_present_terminal_atom():
    atom_names = ["N", "CA", "C", "O", "CB", "CG", "OD2"]
    atom_to_residue = torch.zeros(len(atom_names), dtype=torch.long)
    metadata = build_chi_update_metadata(
        atom_names,
        atom_to_residue,
        ["ASP"],
        num_chi=4,
    )
    assert bool(metadata.valid[0, 1])
    assert metadata.quartet_indices[0, 1, 3].item() == atom_names.index("OD2")
