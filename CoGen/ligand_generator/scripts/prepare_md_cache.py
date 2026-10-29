from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.md_cache import (
    build_ligand_family_features,
    build_protein_atom_features,
    ligand_class_indices,
    load_complex_topology,
    load_ligand_topology,
    protein_aa_indices,
    remap_heavy_ligand,
)
from datasets.pl_data import ProteinLigandData, torchify_dict


CACHE_VERSION = "targetdiff_dynamic_md_cache_v1"


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidate = project_root / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(value)


def _hybridization_from_amber_type(atom_type: str) -> str:
    atom_type = atom_type.strip().lower()
    if atom_type.startswith(("c", "n", "o", "p", "s")) and (
        atom_type in {"c", "c1", "c2", "n", "n1", "n2", "o", "p", "p2", "s"}
        or len(atom_type) == 2
    ):
        return "SP2"
    return "SP3"


def _build_native_sample(
    row: dict,
    sample_path: Path,
    complex_topology,
    ligand_topology,
) -> ProteinLigandData:
    with np.load(sample_path) as sample:
        protein_pos = np.asarray(sample["protein_pos"], dtype=np.float32)
        ligand_pos_all = np.asarray(sample["ligand_pos"], dtype=np.float32)
        protein_global_indices = np.asarray(
            sample["protein_global_atom_index"],
            dtype=np.int64,
        )
        ligand_global_indices = np.asarray(
            sample["ligand_global_atom_index"],
            dtype=np.int64,
        )
        protein_atom_names = np.asarray(sample["protein_atom_name"])
        protein_residue_names = np.asarray(sample["protein_residue_name"])

    if protein_pos.shape != (protein_global_indices.shape[0], 3):
        raise ValueError(f"Protein coordinate/index mismatch in {sample_path}")
    if ligand_pos_all.shape != (ligand_global_indices.shape[0], 3):
        raise ValueError(f"Ligand coordinate/index mismatch in {sample_path}")
    if ligand_pos_all.shape[0] != ligand_topology.atomic_numbers.shape[0]:
        raise ValueError(f"Ligand atom count mismatch in {sample_path}")

    protein_atomic_numbers = complex_topology.atomic_numbers[protein_global_indices]
    aa_indices = protein_aa_indices(protein_residue_names)
    protein_features = build_protein_atom_features(
        atomic_numbers=protein_atomic_numbers,
        aa_indices=aa_indices,
        atom_names=protein_atom_names,
    )

    complex_atom_count = complex_topology.atomic_numbers.shape[0]
    ligand_offset = complex_atom_count - ligand_topology.atomic_numbers.shape[0]
    expected_ligand_indices = np.arange(
        ligand_offset,
        complex_atom_count,
        dtype=np.int64,
    )
    if not np.array_equal(ligand_global_indices, expected_ligand_indices):
        raise ValueError(
            f"Ligand ordering is not the complex topology tail in {sample_path}"
        )
    if not np.array_equal(
        complex_topology.atomic_numbers[ligand_global_indices],
        ligand_topology.atomic_numbers,
    ):
        raise ValueError(f"Complex/ligand element mismatch in {sample_path}")

    ligand_classes_all, aromatic_all = ligand_class_indices(
        ligand_topology.atomic_numbers,
        ligand_topology.amber_atom_types,
    )
    (
        ligand_elements,
        ligand_classes,
        ligand_masses,
        ligand_bond_index,
        ligand_bond_type,
    ) = remap_heavy_ligand(ligand_topology)
    heavy_mask = ligand_topology.atomic_numbers != 1
    ligand_pos = ligand_pos_all[heavy_mask]
    aromatic = aromatic_all[heavy_mask]
    if not np.array_equal(ligand_classes, ligand_classes_all[heavy_mask]):
        raise AssertionError("Heavy-atom ligand class remapping is inconsistent")
    ligand_features = build_ligand_family_features(ligand_elements, aromatic)
    center_of_mass = (
        ligand_pos * ligand_masses[:, None]
    ).sum(axis=0) / np.maximum(ligand_masses.sum(), 1e-8)

    protein_dict = {
        "element": protein_atomic_numbers,
        "pos": protein_pos,
        "is_backbone": np.asarray(
            [name.strip().upper() in {"N", "CA", "C", "O"} for name in protein_atom_names],
            dtype=np.bool_,
        ),
        "atom_to_aa_type": aa_indices,
        "atom_name": [str(name).strip() for name in protein_atom_names],
    }
    ligand_dict = {
        "element": ligand_elements,
        "pos": ligand_pos,
        "atom_feature": ligand_features,
        "bond_index": ligand_bond_index,
        "bond_type": ligand_bond_type,
        "center_of_mass": center_of_mass.astype(np.float32),
        "hybridization": [
            _hybridization_from_amber_type(atom_type)
            for atom_type, keep in zip(
                ligand_topology.amber_atom_types,
                heavy_mask.tolist(),
            )
            if keep
        ],
    }
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(protein_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_atom_feature = torch.from_numpy(protein_features)
    data.ligand_atom_feature_full = torch.from_numpy(ligand_classes)
    if ligand_bond_type.size:
        data.ligand_bond_feature = F.one_hot(
            torch.from_numpy(ligand_bond_type - 1),
            num_classes=4,
        ).to(torch.float32)
    else:
        data.ligand_bond_feature = torch.zeros((0, 4), dtype=torch.float32)
    data.protein_global_atom_index = torch.from_numpy(protein_global_indices)
    data.ligand_global_atom_index = torch.from_numpy(
        ligand_global_indices[heavy_mask]
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=Path("data/md300/selected_frames_3000.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/md300/prepared_targetdiff_md300_10frames.pt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/md300/cache_build_report.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    rows = _read_rows(args.frames_csv)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows found in {args.frames_csv}")

    complex_cache = {}
    ligand_cache = {}
    samples: list[dict] = []
    records: list[dict] = []
    indices_by_split: dict[str, list[int]] = {}
    rejected = Counter()

    for index, row in enumerate(rows):
        try:
            sample_path = _resolve_path(row["sample_path"], project_root)
            topology_key = (row["topology_path"], row["directory"])
            if topology_key not in complex_cache:
                complex_cache[topology_key] = load_complex_topology(
                    row["topology_path"]
                )
                ligand_cache[topology_key] = load_ligand_topology(
                    Path(row["directory"]) / "ligand.prmtop"
                )
            data = _build_native_sample(
                row=row,
                sample_path=sample_path,
                complex_topology=complex_cache[topology_key],
                ligand_topology=ligand_cache[topology_key],
            )
            sample_index = len(samples)
            samples.append(data.to_dict())
            indices_by_split.setdefault(row["split"], []).append(sample_index)
            records.append(
                {
                    "sample_index": sample_index,
                    "trajectory_id": row["trajectory_id"],
                    "pdb_id": row["pdb_id"],
                    "split": row["split"],
                    "cluster_id": int(row["cluster_id"]),
                    "cluster_frame": int(row["cluster_frame"]),
                    "source_sample_path": str(sample_path),
                    "protein_atom_count": int(data.protein_pos.shape[0]),
                    "ligand_atom_count": int(data.ligand_pos.shape[0]),
                }
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rejected[reason] += 1
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(
                f"processed {index + 1}/{len(rows)} frame rows; "
                f"cache_samples={len(samples)}; rejected={sum(rejected.values())}",
                flush=True,
            )

    metadata = {
        "cache_version": CACHE_VERSION,
        "source_frames_csv": str(args.frames_csv),
        "input_frame_count": len(rows),
        "prepared_sample_count": len(samples),
        "rejected_sample_count": sum(rejected.values()),
        "feature_contract": {
            "protein_atom_feature_dim": 27,
            "protein_element_classes": [1, 6, 7, 8, 16, 34],
            "ligand_atom_class_dim": 13,
            "ligand_atom_mode": "add_aromatic",
            "ligand_hydrogens_removed": True,
            "pocket_cutoff_angstrom": 12.0,
        },
        "rejection_reasons": dict(rejected),
        "split_counts": {
            split: len(indices) for split, indices in indices_by_split.items()
        },
        "trajectory_count_with_samples": len(
            {record["trajectory_id"] for record in records}
        ),
    }
    payload = {
        "metadata": metadata,
        "samples": samples,
        "records": records,
        "indices_by_split": indices_by_split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.report.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
