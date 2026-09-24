"""Raw Apo2Mol PDB/SDF adapter for the new PocketDiff schema.

The adapter deliberately sits before graph construction.  It reads the pocket
files supplied by Apo2Mol, establishes a deterministic apo/holo atom
correspondence, aligns holo coordinates (and the ligand) into the apo frame,
and returns a :class:`~pocketdiff.data.schema.PocketComplex`.

The released Apo2Mol pocket files expose a packaging quirk: the two structures
often use different chain, segment, or residue-number labels even though their
records are paired in the same file order.  We therefore use file order as the
correspondence *only after* checking equal length and matching
``(residue_name, atom_name, element)`` for every paired atom.  We do not take an
intersection, sort by coordinates, or truncate.  The holo-side labels become
the normalized canonical labels; raw-label differences are retained in
``AdapterDiagnostics`` so they remain auditable.  A residue-name/atom-signature
mismatch rejects the sample because the frozen schema has one residue identity
for both states.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .schema import PocketComplex
from pocketdiff.geometry.chi import extract_chi_angles
from pocketdiff_v4.constants import TARGETDIFF_RESIDUE_IDS, TARGETDIFF_RESIDUE_NAMES


# Keep these IDs identical to the official TargetDiff definitions without
# importing or modifying the upstream checkout.
AA_NAME_NUMBER: Mapping[str, int] = TARGETDIFF_RESIDUE_IDS
AA_NAMES: Tuple[str, ...] = TARGETDIFF_RESIDUE_NAMES
PROTEIN_ELEMENT_ORDER: Tuple[int, ...] = (1, 6, 7, 8, 16, 34)  # H C N O S Se
BACKBONE_NAMES = frozenset(("CA", "C", "N", "O"))

# TargetDiff's hard ``add_aromatic`` ligand vocabulary.  Unknown types are
# rejected rather than silently converted to hydrogen.
LIGAND_TYPE_MAP: Mapping[Tuple[int, bool], int] = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12,
}

# This is a policy identifier, not the selected altloc character.  It makes
# the normalized key explicit and stable across apo/holo files.
ALTLOC_POLICY = "highest_occupancy_then_lexicographic"

_ELEMENT_SYMBOL_TO_Z: Mapping[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
    "Se": 34,
}


class AdapterError(ValueError):
    """Base error for a sample that cannot satisfy the Phase 1 contract."""

    def __init__(self, message: str, *, reason_code: str = "adapter_error") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class InputFileError(AdapterError):
    """A required raw Apo2Mol file is missing or malformed."""


class AtomIdentityMismatchError(AdapterError):
    """Apo and holo atom records cannot be paired without guessing."""


class AlignmentError(AdapterError):
    """The rigid alignment is underdetermined or outside the filter bound."""


class UnsupportedLigandAtomTypeError(AdapterError):
    """An RDKit ligand atom is outside the frozen 13-class vocabulary."""


@dataclass(frozen=True)
class AdapterDiagnostics:
    """Auditable measurements for one successful conversion."""

    sample_id: str
    num_protein_atoms: int
    num_residues: int
    num_ligand_atoms: int
    num_calpha: int
    aligned_calpha_rmsd: float
    rotation_determinant: float
    raw_key_mismatch_count: int
    raw_chain_mismatch_count: int
    raw_segment_mismatch_count: int
    raw_sequence_mismatch_count: int
    raw_insertion_mismatch_count: int
    normalized_by_paired_order: bool

    def to_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        return asdict(self)


@dataclass(frozen=True)
class _PDBAtom:
    serial: int
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_sequence_id: str
    insertion_code: str
    segment_id: str
    element: str
    atomic_number: int
    occupancy: float
    position: np.ndarray
    source_index: int

    @property
    def source_key(self) -> Tuple[str, ...]:
        return (
            self.chain_id,
            self.segment_id,
            self.residue_sequence_id,
            self.insertion_code,
            self.residue_name,
            self.atom_name,
            self.element,
            self.altloc,
        )

    @property
    def biological_signature(self) -> Tuple[str, str, int]:
        return (self.residue_name, self.atom_name, self.atomic_number)


def _clean_element(raw: str, atom_name: str) -> str:
    """Return a PDB element symbol, using atom-name inference only if needed."""

    raw = raw.strip()
    if raw:
        return raw[0].upper() + raw[1:].lower()

    # PDB atom names may be right aligned for one-letter elements.  Apo2Mol
    # normally supplies element columns, but this fallback keeps the parser
    # deterministic for hand-written smoke fixtures.
    token = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not token:
        raise InputFileError(
            f"cannot infer an element for atom {atom_name!r}",
            reason_code="missing_element",
        )
    return token[0].upper() + token[1:].lower()


def _parse_pdb_atoms(path: Path, *, include_hydrogen: bool = True) -> List[_PDBAtom]:
    """Parse ATOM records and select altlocs without reordering coordinates.

    ``include_hydrogen=False`` is used by the raw Apo2Mol adapter.  The
    diffusion state and χ topology operate on the canonical protein heavy-atom
    inventory; keeping explicit PDB hydrogens would make otherwise valid
    samples fail topology construction and would also make apo/holo atom
    counts depend on the source writer.  The default remains ``True`` for the
    low-level parser so callers that need to audit raw records keep the full
    input.
    """

    if not path.is_file():
        raise InputFileError(f"PDB file does not exist: {path}", reason_code="missing_pdb")

    atoms: List[_PDBAtom] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise InputFileError(f"cannot read PDB file {path}: {exc}", reason_code="read_pdb") from exc

    for source_index, line in enumerate(lines):
        record = line[0:6].strip().upper()
        if record == "ENDMDL":
            break
        if record != "ATOM":
            continue
        if len(line) < 54:
            raise InputFileError(
                f"short ATOM record at {path}:{source_index + 1}",
                reason_code="malformed_pdb",
            )
        try:
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip().upper()
            chain_id = line[21:22].strip()
            residue_sequence_id = line[22:26].strip()
            insertion_code = line[26:27].strip()
            segment_id = line[72:76].strip() if len(line) >= 76 else ""
            altloc = line[16:17].strip()
            element = _clean_element(line[76:78] if len(line) >= 78 else "", atom_name)
            atomic_number = _ELEMENT_SYMBOL_TO_Z.get(element)
            if atomic_number is None:
                raise InputFileError(
                    f"unsupported protein element {element!r} at {path}:{source_index + 1}",
                    reason_code="unsupported_protein_element",
                )
            occupancy_field = line[54:60].strip()
            occupancy = float(occupancy_field) if occupancy_field else 0.0
            position = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
            if not np.isfinite(position).all() or not np.isfinite(occupancy):
                raise InputFileError(
                    f"non-finite ATOM value at {path}:{source_index + 1}",
                    reason_code="nonfinite_pdb",
                )
            serial = int(line[6:11])
        except AdapterError:
            raise
        except (TypeError, ValueError) as exc:
            raise InputFileError(
                f"malformed ATOM record at {path}:{source_index + 1}: {line!r}",
                reason_code="malformed_pdb",
            ) from exc

        atom = _PDBAtom(
            serial=serial,
            atom_name=atom_name,
            altloc=altloc,
            residue_name=residue_name,
            chain_id=chain_id,
            residue_sequence_id=residue_sequence_id,
            insertion_code=insertion_code,
            segment_id=segment_id,
            element=element,
            atomic_number=atomic_number,
            occupancy=occupancy,
            position=position,
            source_index=source_index,
        )
        if include_hydrogen or atom.atomic_number != 1:
            atoms.append(atom)

    if not atoms:
        raise InputFileError(f"PDB contains no ATOM records: {path}", reason_code="empty_pdb")

    # Group only records that describe the same atom apart from altloc.  The
    # selected records are returned in their original file order; no geometric
    # or lexical sorting is used to establish correspondence.
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[_PDBAtom]] = {}
    for atom in atoms:
        key = (
            atom.chain_id,
            atom.segment_id,
            atom.residue_sequence_id,
            atom.insertion_code,
            atom.residue_name,
            atom.atom_name,
            atom.element,
        )
        groups.setdefault(key, []).append(atom)

    selected: List[_PDBAtom] = []
    for candidates in groups.values():
        chosen = sorted(candidates, key=lambda value: (-value.occupancy, value.altloc, value.source_index))[0]
        selected.append(chosen)
    selected.sort(key=lambda value: value.source_index)
    return selected


def _resolve_path(path: Union[str, Path], data_root: Optional[Path]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() and data_root is not None:
        candidate = data_root / candidate
    return candidate


def _parse_ligand(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a 3-D ligand and map it to the explicit 13-class vocabulary."""

    if not path.is_file():
        raise InputFileError(f"ligand file does not exist: {path}", reason_code="missing_ligand")
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - exercised only in a bad env
        raise InputFileError(
            "RDKit is required for Apo2Mol SDF parsing",
            reason_code="missing_rdkit",
        ) from exc

    try:
        molecule = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
        if molecule is None:
            raise ValueError("RDKit returned None")
        Chem.SanitizeMol(molecule)
        molecule = Chem.RemoveHs(molecule)
    except Exception as exc:
        raise InputFileError(f"cannot sanitize ligand {path}: {exc}", reason_code="malformed_ligand") from exc

    if molecule.GetNumConformers() == 0:
        raise InputFileError(f"ligand has no 3-D conformer: {path}", reason_code="missing_ligand_conformer")
    if molecule.GetNumAtoms() == 0:
        raise InputFileError(f"ligand has no atoms after hydrogen removal: {path}", reason_code="empty_ligand")

    conformer = molecule.GetConformer(0)
    positions = np.empty((molecule.GetNumAtoms(), 3), dtype=np.float64)
    ligand_types = np.empty((molecule.GetNumAtoms(),), dtype=np.int64)
    for index, atom in enumerate(molecule.GetAtoms()):
        atomic_number = int(atom.GetAtomicNum())
        aromatic = bool(atom.GetIsAromatic())
        type_key = (atomic_number, aromatic)
        if type_key not in LIGAND_TYPE_MAP:
            raise UnsupportedLigandAtomTypeError(
                f"unsupported ligand atom at {path}:{index}: Z={atomic_number}, aromatic={aromatic}",
                reason_code="unsupported_ligand_atom_type",
            )
        point = conformer.GetAtomPosition(index)
        positions[index] = (float(point.x), float(point.y), float(point.z))
        ligand_types[index] = LIGAND_TYPE_MAP[type_key]

    if not np.isfinite(positions).all():
        raise InputFileError(f"ligand has non-finite coordinates: {path}", reason_code="nonfinite_ligand")
    return positions, ligand_types


def _kabsch_holo_to_apo(
    holo_ca: np.ndarray,
    apo_ca: np.ndarray,
    *,
    max_rmsd: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return row-vector ``(rotation, translation, RMSD)`` for holo → apo."""

    if holo_ca.shape != apo_ca.shape or holo_ca.ndim != 2 or holo_ca.shape[1] != 3:
        raise AlignmentError("invalid C-alpha arrays for Kabsch alignment", reason_code="invalid_alignment_input")
    if holo_ca.shape[0] < 3:
        raise AlignmentError(
            f"at least 3 matched C-alpha atoms are required, got {holo_ca.shape[0]}",
            reason_code="too_few_calpha",
        )

    holo_center = holo_ca.mean(axis=0)
    apo_center = apo_ca.mean(axis=0)
    holo_centered = holo_ca - holo_center
    apo_centered = apo_ca - apo_center
    if np.linalg.matrix_rank(holo_centered, tol=1e-6) < 2 or np.linalg.matrix_rank(apo_centered, tol=1e-6) < 2:
        raise AlignmentError(
            "matched C-alpha atoms are collinear or coincident",
            reason_code="degenerate_alignment_geometry",
        )

    covariance = holo_centered.T @ apo_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt  # row-vector convention: holo @ rotation + translation
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = u @ vt
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not np.isfinite(determinant):
        raise AlignmentError("Kabsch failed to produce a proper rotation", reason_code="improper_rotation")

    translation = apo_center - holo_center @ rotation
    aligned = holo_ca @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - apo_ca) ** 2, axis=1))))
    if not np.isfinite(rmsd):
        raise AlignmentError("aligned C-alpha RMSD is non-finite", reason_code="nonfinite_alignment")
    if rmsd > max_rmsd:
        raise AlignmentError(
            f"aligned C-alpha RMSD {rmsd:.4f} exceeds {max_rmsd:.4f} Å",
            reason_code="alignment_rmsd_too_large",
        )
    return rotation, translation, rmsd


def _make_protein_features(elements: Sequence[int], residue_types: Sequence[int], atom_names: Sequence[str]) -> torch.Tensor:
    element_tensor = torch.tensor(elements, dtype=torch.long)
    residue_tensor = torch.tensor(residue_types, dtype=torch.long)
    element_hot = (element_tensor[:, None] == torch.tensor(PROTEIN_ELEMENT_ORDER, dtype=torch.long)[None, :]).to(torch.float32)
    residue_hot = F.one_hot(residue_tensor, num_classes=20).to(torch.float32)
    backbone = torch.tensor(
        [[1.0 if atom_name in BACKBONE_NAMES else 0.0] for atom_name in atom_names],
        dtype=torch.float32,
    )
    return torch.cat((element_hot, residue_hot, backbone), dim=1)


class Apo2MolAdapter:
    """Convert raw Apo2Mol records to the independent PocketDiff schema."""

    def __init__(self, data_root: Optional[Union[str, Path]] = None, *, max_aligned_ca_rmsd: float = 4.0) -> None:
        if max_aligned_ca_rmsd <= 0.0:
            raise ValueError("max_aligned_ca_rmsd must be positive")
        self.data_root = Path(data_root) if data_root is not None else None
        self.max_aligned_ca_rmsd = float(max_aligned_ca_rmsd)
        self.last_diagnostics: Optional[AdapterDiagnostics] = None

    def convert_record(self, record: Sequence[object], *, sample_id: Optional[str] = None) -> PocketComplex:
        """Convert the first three fields of an Apo2Mol split tuple."""

        complex_value, diagnostics = self.convert_record_with_diagnostics(record, sample_id=sample_id)
        self.last_diagnostics = diagnostics
        return complex_value

    def convert_record_with_diagnostics(
        self,
        record: Sequence[object],
        *,
        sample_id: Optional[str] = None,
    ) -> Tuple[PocketComplex, AdapterDiagnostics]:
        if len(record) < 3:
            raise InputFileError("Apo2Mol record must contain at least holo pocket, apo pocket, and ligand paths", reason_code="bad_record")
        holo_path = _resolve_path(record[0], self.data_root)
        apo_path = _resolve_path(record[1], self.data_root)
        ligand_path = _resolve_path(record[2], self.data_root)
        if sample_id is None:
            sample_id = Path(str(record[0])).parent.name or Path(str(record[0])).stem
        return self.convert_paths(holo_path, apo_path, ligand_path, sample_id=sample_id)

    def convert_paths(
        self,
        holo_pocket_path: Union[str, Path],
        apo_pocket_path: Union[str, Path],
        ligand_path: Union[str, Path],
        *,
        sample_id: str,
    ) -> Tuple[PocketComplex, AdapterDiagnostics]:
        if not sample_id:
            raise InputFileError("sample_id must be non-empty", reason_code="bad_sample_id")

        # DynamicBind and the released Apo2Mol preprocessing both remove
        # explicit protein hydrogens before geometry/model construction.  The
        # local χ topology has the same heavy-atom contract; filter both sides
        # before correspondence so hydrogen-rich and hydrogen-free writers
        # produce the same canonical protein state.
        holo_atoms = _parse_pdb_atoms(Path(holo_pocket_path), include_hydrogen=False)
        apo_atoms = _parse_pdb_atoms(Path(apo_pocket_path), include_hydrogen=False)
        if len(holo_atoms) != len(apo_atoms):
            raise AtomIdentityMismatchError(
                f"{sample_id}: apo/holo atom counts differ ({len(apo_atoms)} vs {len(holo_atoms)})",
                reason_code="atom_count_mismatch",
            )

        raw_key_mismatch_count = sum(h.source_key[:-1] != a.source_key[:-1] for h, a in zip(holo_atoms, apo_atoms))
        raw_chain_mismatch_count = sum(h.chain_id != a.chain_id for h, a in zip(holo_atoms, apo_atoms))
        raw_segment_mismatch_count = sum(h.segment_id != a.segment_id for h, a in zip(holo_atoms, apo_atoms))
        raw_sequence_mismatch_count = sum(h.residue_sequence_id != a.residue_sequence_id for h, a in zip(holo_atoms, apo_atoms))
        raw_insertion_mismatch_count = sum(h.insertion_code != a.insertion_code for h, a in zip(holo_atoms, apo_atoms))

        mismatches = [
            (index, holo_atom, apo_atom)
            for index, (holo_atom, apo_atom) in enumerate(zip(holo_atoms, apo_atoms))
            if holo_atom.biological_signature != apo_atom.biological_signature
        ]
        if mismatches:
            index, holo_atom, apo_atom = mismatches[0]
            raise AtomIdentityMismatchError(
                f"{sample_id}: paired atom signature differs at index {index}: "
                f"holo=({holo_atom.residue_name},{holo_atom.atom_name},{holo_atom.element}) "
                f"apo=({apo_atom.residue_name},{apo_atom.atom_name},{apo_atom.element}); "
                f"mismatched_records={len(mismatches)}",
                reason_code="atom_signature_mismatch",
            )

        # The holo labels define the normalized canonical order.  Since every
        # pair passed the biological-signature check, this is a one-to-one
        # order-preserving pairing rather than an atom intersection.
        canonical_atoms = holo_atoms
        residue_index: Dict[Tuple[str, str, str, str, str], int] = {}
        residue_chain_id: List[str] = []
        residue_sequence_id: List[str] = []
        residue_names: List[str] = []
        atom_to_residue: List[int] = []
        for atom in canonical_atoms:
            if atom.residue_name not in AA_NAME_NUMBER:
                raise AtomIdentityMismatchError(
                    f"{sample_id}: unsupported amino-acid residue {atom.residue_name!r}",
                    reason_code="unsupported_residue",
                )
            residue_key = (
                atom.chain_id,
                atom.segment_id,
                atom.residue_sequence_id,
                atom.insertion_code,
                atom.residue_name,
            )
            if residue_key not in residue_index:
                residue_index[residue_key] = len(residue_chain_id)
                residue_chain_id.append(atom.chain_id)
                residue_sequence_id.append(atom.residue_sequence_id)
                residue_names.append(atom.residue_name)
            atom_to_residue.append(residue_index[residue_key])

        holo_pos = np.asarray([atom.position for atom in holo_atoms], dtype=np.float64)
        apo_pos = np.asarray([atom.position for atom in apo_atoms], dtype=np.float64)
        ca_indices = [index for index, atom in enumerate(canonical_atoms) if atom.atom_name == "CA"]
        if len(ca_indices) < 3:
            raise AlignmentError(
                f"{sample_id}: at least 3 matched C-alpha atoms are required, got {len(ca_indices)}",
                reason_code="too_few_calpha",
            )
        rotation, translation, aligned_ca_rmsd = _kabsch_holo_to_apo(
            holo_pos[ca_indices],
            apo_pos[ca_indices],
            max_rmsd=self.max_aligned_ca_rmsd,
        )
        ligand_pos, ligand_type = _parse_ligand(Path(ligand_path))

        aligned_holo_pos = holo_pos @ rotation + translation
        aligned_ligand_pos = ligand_pos @ rotation + translation
        center_offset = apo_pos.mean(axis=0)
        centered_apo = apo_pos - center_offset
        centered_holo = aligned_holo_pos - center_offset
        centered_ligand = aligned_ligand_pos - center_offset

        chi_apo, chi_apo_mask = extract_chi_angles(
            torch.from_numpy(centered_apo.astype(np.float32, copy=False)),
            [atom.atom_name for atom in canonical_atoms],
            torch.tensor(atom_to_residue, dtype=torch.long),
            residue_names,
            num_chi=5,
        )
        chi_holo, chi_holo_mask = extract_chi_angles(
            torch.from_numpy(centered_holo.astype(np.float32, copy=False)),
            [atom.atom_name for atom in canonical_atoms],
            torch.tensor(atom_to_residue, dtype=torch.long),
            residue_names,
            num_chi=5,
        )
        chi_mask = chi_apo_mask & chi_holo_mask
        chi_apo = torch.where(chi_mask, chi_apo, torch.zeros_like(chi_apo))
        chi_holo = torch.where(chi_mask, chi_holo, torch.zeros_like(chi_holo))

        residue_type = [AA_NAME_NUMBER[name] for name in residue_names]
        protein_feature = _make_protein_features(
            [atom.atomic_number for atom in canonical_atoms],
            [residue_type[index] for index in atom_to_residue],
            [atom.atom_name for atom in canonical_atoms],
        )

        frame_valid_values: List[bool] = []
        for residue_id in range(len(residue_names)):
            names_holo = {
                atom.atom_name for index, atom in enumerate(holo_atoms) if atom_to_residue[index] == residue_id
            }
            names_apo = {
                atom.atom_name for index, atom in enumerate(apo_atoms) if atom_to_residue[index] == residue_id
            }
            frame_valid_values.append(all(name in names_holo and name in names_apo for name in ("N", "CA", "C")))

        complex_value = PocketComplex(
            sample_id=sample_id,
            protein_pos_apo=torch.from_numpy(centered_apo.astype(np.float32, copy=False)),
            protein_pos_holo=torch.from_numpy(centered_holo.astype(np.float32, copy=False)),
            protein_feature=protein_feature,
            protein_element=torch.tensor([atom.atomic_number for atom in canonical_atoms], dtype=torch.long),
            protein_atom_name=[atom.atom_name for atom in canonical_atoms],
            protein_residue_name=[atom.residue_name for atom in canonical_atoms],
            atom_to_residue=torch.tensor(atom_to_residue, dtype=torch.long),
            residue_type=torch.tensor(residue_type, dtype=torch.long),
            residue_chain_id=residue_chain_id,
            residue_sequence_id=residue_sequence_id,
            frame_valid=torch.tensor(frame_valid_values, dtype=torch.bool),
            chi_apo=chi_apo,
            chi_holo=chi_holo,
            chi_mask=chi_mask,
            ligand_pos_ref=torch.from_numpy(centered_ligand.astype(np.float32, copy=False)),
            ligand_type_ref=torch.from_numpy(ligand_type.astype(np.int64, copy=False)),
            center_offset=torch.from_numpy(center_offset.astype(np.float32, copy=False)),
        )

        diagnostics = AdapterDiagnostics(
            sample_id=sample_id,
            num_protein_atoms=len(canonical_atoms),
            num_residues=len(residue_names),
            num_ligand_atoms=int(ligand_type.shape[0]),
            num_calpha=len(ca_indices),
            aligned_calpha_rmsd=aligned_ca_rmsd,
            rotation_determinant=float(np.linalg.det(rotation)),
            raw_key_mismatch_count=raw_key_mismatch_count,
            raw_chain_mismatch_count=raw_chain_mismatch_count,
            raw_segment_mismatch_count=raw_segment_mismatch_count,
            raw_sequence_mismatch_count=raw_sequence_mismatch_count,
            raw_insertion_mismatch_count=raw_insertion_mismatch_count,
            normalized_by_paired_order=raw_key_mismatch_count > 0,
        )
        return complex_value, diagnostics


def load_apo2mol_record(
    record: Sequence[object],
    *,
    data_root: Optional[Union[str, Path]] = None,
    sample_id: Optional[str] = None,
) -> PocketComplex:
    """Functional convenience wrapper around :class:`Apo2MolAdapter`."""

    return Apo2MolAdapter(data_root).convert_record(record, sample_id=sample_id)


__all__ = [
    "AA_NAME_NUMBER",
    "AdapterDiagnostics",
    "AdapterError",
    "AlignmentError",
    "Apo2MolAdapter",
    "AtomIdentityMismatchError",
    "InputFileError",
    "LIGAND_TYPE_MAP",
    "UnsupportedLigandAtomTypeError",
    "load_apo2mol_record",
]
