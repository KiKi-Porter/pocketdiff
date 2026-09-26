import pickle
from pathlib import Path

import pytest
import torch

from pocketdiff.data.apo2mol_adapter import (
    Apo2MolAdapter,
    AtomIdentityMismatchError,
    _parse_pdb_atoms,
)


DATA_ROOT = Path("Apo2Mol-main/Apo2MOl-dataset/data_folder")
SPLIT_PATH = DATA_ROOT.parent / "split_druglike_dict.pkl"


def _first_train_record():
    if not DATA_ROOT.is_dir() or not SPLIT_PATH.is_file():
        pytest.skip("Apo2Mol raw dataset is not available in this checkout")
    with SPLIT_PATH.open("rb") as handle:
        return pickle.load(handle)["train"][0]


def test_real_apo2mol_record_meets_pocket_complex_contract():
    record = _first_train_record()
    value, diagnostics = Apo2MolAdapter(DATA_ROOT).convert_record_with_diagnostics(record)

    assert value.sample_id == "3txj__1__1.A__1.K"
    assert value.protein_pos_apo.shape == value.protein_pos_holo.shape
    assert value.protein_feature.shape == (value.num_protein_atoms, 27)
    assert value.ligand_pos_ref.shape[1] == 3
    assert value.ligand_type_ref.dtype == torch.long
    assert bool(torch.isfinite(value.protein_pos_apo).all())
    assert bool(torch.isfinite(value.protein_pos_holo).all())
    assert bool(torch.isfinite(value.ligand_pos_ref).all())
    assert bool(torch.allclose(value.protein_pos_apo.mean(dim=0), torch.zeros(3), atol=2e-5))
    assert bool(value.frame_valid.all())
    assert diagnostics.rotation_determinant > 0.0
    assert diagnostics.aligned_calpha_rmsd < 4.0
    # This fixture demonstrates the released-data segment-label mismatch; it
    # must be auditable rather than silently treated as an atom intersection.
    assert diagnostics.normalized_by_paired_order
    assert diagnostics.raw_segment_mismatch_count == value.num_protein_atoms


def test_hydrogen_rich_apo2mol_record_is_canonicalized_to_heavy_atoms():
    if not DATA_ROOT.is_dir() or not SPLIT_PATH.is_file():
        pytest.skip("Apo2Mol raw dataset is not available in this checkout")
    with SPLIT_PATH.open("rb") as handle:
        record = pickle.load(handle)["valid"][4]
    value = Apo2MolAdapter(DATA_ROOT).convert_record(record)

    assert value.num_protein_atoms > 0
    assert all(element != 1 for element in value.protein_element.tolist())
    assert not any(name.startswith("H") for name in value.protein_atom_name)
    assert bool(torch.isfinite(value.protein_pos_apo).all())
    assert bool(torch.isfinite(value.protein_pos_holo).all())


def _pdb_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_id: int,
    x: float,
    occupancy: float,
    *,
    altloc: str = "",
    element: str = "C",
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:>4s}{altloc:1s}{residue_name:>3s} A"
        f"{residue_id:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{occupancy:6.2f}{10.0:6.2f}          {element:>2s}  "
    )


def test_altloc_selection_uses_occupancy_then_lexicographic_tie_break(tmp_path):
    path = tmp_path / "altloc.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_line(1, "CA", "ALA", 1, 1.0, 0.20, altloc="B"),
                _pdb_line(2, "CA", "ALA", 1, 2.0, 0.20, altloc="A"),
                _pdb_line(3, "N", "ALA", 1, 3.0, 1.00),
                "END",
            ]
        )
        + "\n"
    )
    atoms = _parse_pdb_atoms(path)
    assert [atom.atom_name for atom in atoms] == ["CA", "N"]
    assert atoms[0].altloc == "A"
    assert atoms[0].position[0] == pytest.approx(2.0)


def test_residue_signature_mismatch_is_rejected(tmp_path):
    record = _first_train_record()
    holo = DATA_ROOT / record[0]
    apo = DATA_ROOT / record[1]
    ligand = DATA_ROOT / record[2]
    holo_copy = tmp_path / "holo.pdb"
    apo_copy = tmp_path / "apo.pdb"
    ligand_copy = tmp_path / "ligand.sdf"
    holo_copy.write_text(holo.read_text())
    # Replace only the first residue name in the apo copy.  Coordinates and
    # atom order remain unchanged, so the failure is specifically identity.
    apo_lines = apo.read_text().splitlines()
    first_atom = next(index for index, line in enumerate(apo_lines) if line.startswith("ATOM"))
    line = apo_lines[first_atom]
    apo_lines[first_atom] = line[:17] + "GLY" + line[20:]
    apo_copy.write_text("\n".join(apo_lines) + "\n")
    ligand_copy.write_text(ligand.read_text())

    adapter = Apo2MolAdapter()
    with pytest.raises(AtomIdentityMismatchError, match="signature differs"):
        adapter.convert_paths(holo_copy, apo_copy, ligand_copy, sample_id="identity-mismatch")
