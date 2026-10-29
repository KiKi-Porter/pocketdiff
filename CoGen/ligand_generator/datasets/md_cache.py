from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.md_utils import _read_prmtop_flag, _topology_atom_count


TARGETDIFF_PROTEIN_ELEMENTS = np.asarray([1, 6, 7, 8, 16, 34], dtype=np.int64)
TARGETDIFF_AA_NAMES = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
)
TARGETDIFF_AA_TO_INDEX = {
    name: index for index, name in enumerate(TARGETDIFF_AA_NAMES)
}
PROTEIN_AA_ALIASES = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
}

TARGETDIFF_LIGAND_CLASS = {
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

# GAFF/Amber aromatic atom types used by the ligand prmtop files.
AROMATIC_AMBER_TYPES = {
    "ca",
    "cc",
    "cd",
    "ce",
    "cf",
    "cg",
    "ch",
    "cp",
    "cq",
    "cu",
    "cv",
    "cw",
    "cx",
    "cy",
    "cz",
    "na",
    "nb",
    "nc",
    "nd",
    "ne",
    "nf",
    "ng",
    "nh",
    "ni",
    "nj",
    "nk",
    "nl",
    "nm",
    "nn",
    "no",
    "np",
    "nq",
    "nr",
    "ns",
    "nt",
    "nu",
    "nv",
    "nw",
    "nx",
    "ny",
    "nz",
}


@dataclass(frozen=True)
class ComplexTopology:
    atomic_numbers: np.ndarray
    atom_names: tuple[str, ...]


@dataclass(frozen=True)
class LigandTopology:
    atomic_numbers: np.ndarray
    atom_names: tuple[str, ...]
    amber_atom_types: tuple[str, ...]
    masses: np.ndarray
    bond_index: np.ndarray
    bond_type: np.ndarray


def load_complex_topology(path: str | Path) -> ComplexTopology:
    path = Path(path)
    atomic_numbers = np.asarray(
        [int(value) for value in _read_prmtop_flag(path, "ATOMIC_NUMBER")],
        dtype=np.int64,
    )
    atom_names = tuple(_read_prmtop_flag(path, "ATOM_NAME"))
    expected = _topology_atom_count(path)
    if len(atomic_numbers) != expected or len(atom_names) != expected:
        raise ValueError(
            f"Topology atom count mismatch in {path}: "
            f"atomic_numbers={len(atomic_numbers)}, atom_names={len(atom_names)}, "
            f"expected={expected}"
        )
    return ComplexTopology(atomic_numbers=atomic_numbers, atom_names=atom_names)


def _parse_bonds(path: Path, atom_count: int) -> tuple[np.ndarray, np.ndarray]:
    edges: list[tuple[int, int, int]] = []
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = [int(value) for value in _read_prmtop_flag(path, flag)]
        if len(values) % 3 != 0:
            raise ValueError(f"{flag} is not a triplet list in {path}")
        for index in range(0, len(values), 3):
            source = values[index] // 3
            target = values[index + 1] // 3
            if source == target:
                continue
            if not (0 <= source < atom_count and 0 <= target < atom_count):
                raise ValueError(
                    f"Invalid bond ({source}, {target}) for {atom_count} atoms in {path}"
                )
            # The Amber topology stores a parameter index, not a chemical bond
            # order. The graph only needs connectivity here; aromatic edges are
            # marked separately from the atom types.
            edges.append((source, target, 1))
            edges.append((target, source, 1))
    if not edges:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    edge_array = np.asarray(edges, dtype=np.int64)
    return edge_array[:, :2].T, edge_array[:, 2]


def load_ligand_topology(path: str | Path) -> LigandTopology:
    path = Path(path)
    atom_count = _topology_atom_count(path)
    atomic_numbers = np.asarray(
        [int(value) for value in _read_prmtop_flag(path, "ATOMIC_NUMBER")],
        dtype=np.int64,
    )
    atom_names = tuple(_read_prmtop_flag(path, "ATOM_NAME"))
    amber_atom_types = tuple(
        value.strip().lower() for value in _read_prmtop_flag(path, "AMBER_ATOM_TYPE")
    )
    masses = np.asarray(
        [float(value) for value in _read_prmtop_flag(path, "MASS")],
        dtype=np.float32,
    )
    if not (
        len(atomic_numbers)
        == len(atom_names)
        == len(amber_atom_types)
        == len(masses)
        == atom_count
    ):
        raise ValueError(f"Ligand topology fields have inconsistent sizes in {path}")
    bond_index, bond_type = _parse_bonds(path, atom_count)
    return LigandTopology(
        atomic_numbers=atomic_numbers,
        atom_names=atom_names,
        amber_atom_types=amber_atom_types,
        masses=masses,
        bond_index=bond_index,
        bond_type=bond_type,
    )


def normalize_protein_residue_name(name: str) -> str:
    normalized = name.strip().upper()
    return PROTEIN_AA_ALIASES.get(normalized, normalized)


def protein_aa_indices(residue_names: np.ndarray) -> np.ndarray:
    indices = []
    for name in residue_names.tolist():
        normalized = normalize_protein_residue_name(str(name))
        if normalized not in TARGETDIFF_AA_TO_INDEX:
            raise ValueError(
                f"Residue {name!r} cannot be represented by TargetDiff's 20 AA classes"
            )
        indices.append(TARGETDIFF_AA_TO_INDEX[normalized])
    return np.asarray(indices, dtype=np.int64)


def is_aromatic_amber_type(atom_type: str) -> bool:
    return atom_type.strip().lower() in AROMATIC_AMBER_TYPES


def ligand_class_indices(
    atomic_numbers: np.ndarray,
    amber_atom_types: tuple[str, ...] | list[str],
) -> tuple[np.ndarray, np.ndarray]:
    aromatic = np.asarray(
        [is_aromatic_amber_type(value) for value in amber_atom_types],
        dtype=np.bool_,
    )
    classes = []
    for atomic_number, is_aromatic in zip(atomic_numbers.tolist(), aromatic.tolist()):
        key = (int(atomic_number), bool(is_aromatic))
        if key not in TARGETDIFF_LIGAND_CLASS:
            raise ValueError(
                f"Element/aromatic pair {key} is outside TargetDiff's 13-class "
                "ligand representation"
            )
        classes.append(TARGETDIFF_LIGAND_CLASS[key])
    return np.asarray(classes, dtype=np.int64), aromatic


def remap_heavy_ligand(
    ligand: LigandTopology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heavy_mask = ligand.atomic_numbers != 1
    old_to_new = np.full(ligand.atomic_numbers.shape[0], -1, dtype=np.int64)
    old_to_new[heavy_mask] = np.arange(int(heavy_mask.sum()), dtype=np.int64)
    edge_index = ligand.bond_index
    keep_edges = heavy_mask[edge_index[0]] & heavy_mask[edge_index[1]]
    remapped_edges = old_to_new[edge_index[:, keep_edges]]
    bond_type = ligand.bond_type[keep_edges].copy()
    classes, aromatic = ligand_class_indices(
        ligand.atomic_numbers,
        ligand.amber_atom_types,
    )
    aromatic_edges = aromatic[edge_index[0, keep_edges]] & aromatic[
        edge_index[1, keep_edges]
    ]
    bond_type[aromatic_edges] = 4
    return (
        ligand.atomic_numbers[heavy_mask],
        classes[heavy_mask],
        ligand.masses[heavy_mask],
        remapped_edges,
        bond_type,
    )


def build_protein_atom_features(
    atomic_numbers: np.ndarray,
    aa_indices: np.ndarray,
    atom_names: np.ndarray,
) -> np.ndarray:
    if np.any(~np.isin(atomic_numbers, TARGETDIFF_PROTEIN_ELEMENTS)):
        unsupported = sorted(
            set(atomic_numbers.tolist()) - set(TARGETDIFF_PROTEIN_ELEMENTS.tolist())
        )
        raise ValueError(f"Protein elements {unsupported} are outside TargetDiff 27D input")
    element = (atomic_numbers[:, None] == TARGETDIFF_PROTEIN_ELEMENTS[None, :]).astype(
        np.float32
    )
    amino_acid = np.eye(len(TARGETDIFF_AA_NAMES), dtype=np.float32)[aa_indices]
    backbone = np.asarray(
        [name.strip().upper() in {"N", "CA", "C", "O"} for name in atom_names],
        dtype=np.float32,
    )[:, None]
    return np.concatenate([element, amino_acid, backbone], axis=-1)


def build_ligand_family_features(
    atomic_numbers: np.ndarray,
    aromatic: np.ndarray,
) -> np.ndarray:
    # TargetDiff only reads the Aromatic column from this auxiliary matrix
    # before constructing the 13-class categorical input.
    features = np.zeros((atomic_numbers.shape[0], 8), dtype=np.int64)
    features[:, 2] = aromatic.astype(np.int64)
    return features
