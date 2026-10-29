from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import netcdf_file


PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}
SOLVENT_OR_ION_RESIDUES = {
    "WAT", "HOH", "TIP3", "TIP3P", "Na+", "Cl-", "NA", "CL", "K+", "K",
    "MG", "CA", "ZN",
}


@dataclass(frozen=True)
class TopologyInfo:
    atom_names: tuple[str, ...]
    residue_names: tuple[str, ...]
    residue_pointers: tuple[int, ...]
    protein_atom_indices: np.ndarray
    protein_ca_indices: np.ndarray
    ligand_atom_indices: np.ndarray
    ligand_residue_names: tuple[str, ...]
    protein_residue_ranges: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    pdb_id: str
    directory: str
    trajectory_path: str
    topology_path: str
    reference_path: str
    n_frames: int
    n_atoms: int
    n_protein_atoms: int
    n_ligand_atoms: int
    ligand_residues: tuple[str, ...]
    n_protein_ca: int
    nc_bytes: int
    protein_motion_rmsd: float
    ligand_relative_motion: float


def _read_prmtop_flag(path: Path, flag: str) -> list[str]:
    lines = path.read_text(errors="ignore").splitlines()
    target = f"%FLAG {flag}"
    start = next(i for i, line in enumerate(lines) if line.strip() == target)
    values: list[str] = []
    index = start + 2
    width = 4 if flag in {"ATOM_NAME", "RESIDUE_LABEL", "AMBER_ATOM_TYPE"} else None
    while index < len(lines) and not lines[index].strip().startswith("%FLAG "):
        line = lines[index]
        if width is None:
            values.extend(line.split())
        else:
            values.extend(line[offset:offset + width].strip() for offset in range(0, len(line), width))
        index += 1
    return values


def _topology_atom_count(path: str | Path) -> int:
    values = _read_prmtop_flag(Path(path), "POINTERS")
    if not values:
        raise ValueError(f"Missing POINTERS in {path}")
    return int(values[0])


def load_topology(path: str | Path, ligand_path: str | Path | None = None) -> TopologyInfo:
    path = Path(path)
    atom_names = tuple(_read_prmtop_flag(path, "ATOM_NAME"))
    residue_names = tuple(_read_prmtop_flag(path, "RESIDUE_LABEL"))
    residue_pointers = tuple(int(v) for v in _read_prmtop_flag(path, "RESIDUE_POINTER"))
    if not atom_names or not residue_names or not residue_pointers:
        raise ValueError(f"Incomplete topology fields in {path}")
    if len(atom_names) != _topology_atom_count(path):
        raise ValueError(
            f"ATOM_NAME count {len(atom_names)} does not match NATOM "
            f"{_topology_atom_count(path)} in {path}"
        )

    residue_atom_ranges = []
    for residue_index, start_one_based in enumerate(residue_pointers):
        start = start_one_based - 1
        end = residue_pointers[residue_index + 1] - 1 if residue_index + 1 < len(residue_pointers) else len(atom_names)
        residue_atom_ranges.append((start, end))

    ligand_count = _topology_atom_count(ligand_path) if ligand_path is not None else 0
    if ligand_count <= 0 or ligand_count >= len(atom_names):
        ligand_count = 0
    protein_atom_limit = len(atom_names) - ligand_count
    protein_indices: list[int] = []
    ca_indices: list[int] = []
    protein_residue_ranges: list[tuple[int, int, str]] = []
    nonprotein_groups: list[tuple[int, str, list[int]]] = []
    for residue_index, residue_name in enumerate(residue_names):
        start, end = residue_atom_ranges[residue_index]
        indices = list(range(start, end))
        normalized_name = residue_name.strip()
        if normalized_name not in SOLVENT_OR_ION_RESIDUES and start < protein_atom_limit:
            indices = [index for index in indices if index < protein_atom_limit]
            protein_indices.extend(indices)
            ca_indices.extend(i for i in indices if atom_names[i].strip() == "CA")
            protein_residue_ranges.append((start, min(end, protein_atom_limit), normalized_name))
        elif normalized_name not in SOLVENT_OR_ION_RESIDUES:
            nonprotein_groups.append((residue_index, normalized_name, indices))

    if ligand_count > 0:
        ligand_indices = list(range(protein_atom_limit, len(atom_names)))
        ligand_names = (residue_names[-1].strip(),)
    elif not nonprotein_groups:
        raise ValueError(f"No non-protein ligand residue found in {path}")
    else:
        # Fallback for a topology without a separate ligand.prmtop.
        max_atoms = max(len(indices) for _, _, indices in nonprotein_groups)
        ligand_groups = [group for group in nonprotein_groups if len(group[2]) == max_atoms]
        ligand_indices = [atom for _, _, indices in ligand_groups for atom in indices]
        ligand_names = tuple(name for _, name, _ in ligand_groups)

    return TopologyInfo(
        atom_names=atom_names,
        residue_names=residue_names,
        residue_pointers=residue_pointers,
        protein_atom_indices=np.asarray(protein_indices, dtype=np.int64),
        protein_ca_indices=np.asarray(ca_indices, dtype=np.int64),
        ligand_atom_indices=np.asarray(ligand_indices, dtype=np.int64),
        ligand_residue_names=ligand_names,
        protein_residue_ranges=tuple(protein_residue_ranges),
    )


def _residue_atom_count(pointers: tuple[int, ...], n_atoms: int, index: int) -> int:
    return (pointers[index + 1] - pointers[index]) if index + 1 < len(pointers) else n_atoms + 1 - pointers[index]


def trajectory_records(root: str | Path) -> list[TrajectoryRecord]:
    root = Path(root)
    records: list[TrajectoryRecord] = []
    for pdb_dir_entry in sorted(root.iterdir()):
        if not pdb_dir_entry.is_dir() or pdb_dir_entry.name == "JSON":
            continue
        for complex_entry in sorted(pdb_dir_entry.iterdir()):
            if not complex_entry.is_dir():
                continue
            trajectory_path = complex_entry / "md_dry.nc"
            topology_path = complex_entry / "complex.prmtop"
            reference_path = complex_entry / "complex_reference.pdb"
            ligand_topology_path = complex_entry / "ligand.prmtop"
            if not (
                trajectory_path.is_file()
                and topology_path.is_file()
                and reference_path.is_file()
                and ligand_topology_path.is_file()
            ):
                continue
            try:
                topology = load_topology(topology_path, ligand_topology_path)
                with netcdf_file(str(trajectory_path), mmap=True) as nc:
                    shape = tuple(int(v) for v in nc.variables["coordinates"].shape)
                    n_frames, n_atoms = shape[:2]
                    cell_lengths = nc.variables.get("cell_lengths")
                    first = np.array(nc.variables["coordinates"][0], dtype=np.float32, copy=True)
                    middle = np.array(nc.variables["coordinates"][n_frames // 2], dtype=np.float32, copy=True)
                    last = np.array(nc.variables["coordinates"][n_frames - 1], dtype=np.float32, copy=True)
                    if cell_lengths is not None:
                        first_box = np.array(cell_lengths[0], dtype=np.float32, copy=True)
                        middle_box = np.array(cell_lengths[n_frames // 2], dtype=np.float32, copy=True)
                        last_box = np.array(cell_lengths[n_frames - 1], dtype=np.float32, copy=True)
                        middle = _unwrap_against_reference(middle, first, middle_box)
                        last = _unwrap_against_reference(last, first, last_box)
                    motion_rmsd = _aligned_ca_rmsd(
                        first, middle, last, topology.protein_ca_indices
                    )
                    ligand_motion = _aligned_ligand_motion(
                        first, middle, last, topology.protein_ca_indices, topology.ligand_atom_indices
                    )
                records.append(TrajectoryRecord(
                    trajectory_id=complex_entry.name,
                    pdb_id=pdb_dir_entry.name,
                    directory=str(complex_entry),
                    trajectory_path=str(trajectory_path),
                    topology_path=str(topology_path),
                    reference_path=str(reference_path),
                    n_frames=n_frames,
                    n_atoms=n_atoms,
                    n_protein_atoms=int(topology.protein_atom_indices.size),
                    n_ligand_atoms=int(topology.ligand_atom_indices.size),
                    ligand_residues=topology.ligand_residue_names,
                    n_protein_ca=int(topology.protein_ca_indices.size),
                    nc_bytes=trajectory_path.stat().st_size,
                    protein_motion_rmsd=motion_rmsd,
                    ligand_relative_motion=ligand_motion,
                ))
            except Exception as exc:
                print(f"[skip] {complex_entry}: {type(exc).__name__}: {exc}")
    return records


def _unwrap_against_reference(frame: np.ndarray, reference: np.ndarray, box: np.ndarray) -> np.ndarray:
    box = np.where(np.abs(box) > 1e-6, box, 1.0)
    delta = frame - reference
    delta = delta - np.round(delta / box[None, :]) * box[None, :]
    return reference + delta


def _kabsch_transform(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_center = reference.mean(axis=0)
    moving_center = moving.mean(axis=0)
    ref_centered = reference - ref_center
    moving_centered = moving - moving_center
    covariance = moving_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = ref_center - moving_center @ rotation
    return rotation, translation


def _aligned_ca_rmsd(
    first: np.ndarray, middle: np.ndarray, last: np.ndarray, ca_indices: np.ndarray
) -> float:
    if ca_indices.size < 3:
        return 0.0
    reference = first[ca_indices]
    values = []
    for frame in (middle, last):
        rotation, translation = _kabsch_transform(reference, frame[ca_indices])
        aligned = frame[ca_indices] @ rotation + translation
        values.append(float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=-1)))))
    return float(np.mean(values))


def _aligned_ligand_motion(
    first: np.ndarray,
    middle: np.ndarray,
    last: np.ndarray,
    ca_indices: np.ndarray,
    ligand_indices: np.ndarray,
) -> float:
    if ca_indices.size < 3 or ligand_indices.size == 0:
        return 0.0
    reference_ca = first[ca_indices]
    reference_centroid = first[ligand_indices].mean(axis=0)
    values = []
    for frame in (middle, last):
        rotation, translation = _kabsch_transform(reference_ca, frame[ca_indices])
        centroid = frame[ligand_indices].mean(axis=0) @ rotation + translation
        values.append(float(np.linalg.norm(centroid - reference_centroid)))
    return float(np.mean(values))


def load_frame(trajectory_path: str | Path, frame_index: int) -> np.ndarray:
    with netcdf_file(str(trajectory_path), mmap=True) as nc:
        return np.array(nc.variables["coordinates"][frame_index], dtype=np.float32, copy=True)


def aligned_frame(
    frame: np.ndarray,
    reference: np.ndarray,
    ca_indices: np.ndarray,
    cell_lengths: np.ndarray | None = None,
) -> np.ndarray:
    if cell_lengths is not None:
        frame = _unwrap_against_reference(frame, reference, cell_lengths)
    if ca_indices.size < 3:
        return frame
    rotation, translation = _kabsch_transform(reference[ca_indices], frame[ca_indices])
    return frame @ rotation + translation
