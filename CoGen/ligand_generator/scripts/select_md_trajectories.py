from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from datasets.md_utils import TrajectoryRecord, trajectory_records


SEED = 20261001


def _record_row(record: TrajectoryRecord) -> dict:
    return {
        "trajectory_id": record.trajectory_id,
        "pdb_id": record.pdb_id,
        "directory": record.directory,
        "trajectory_path": record.trajectory_path,
        "topology_path": record.topology_path,
        "reference_path": record.reference_path,
        "n_frames": record.n_frames,
        "n_atoms": record.n_atoms,
        "n_protein_atoms": record.n_protein_atoms,
        "n_ligand_atoms": record.n_ligand_atoms,
        "ligand_residues": ";".join(record.ligand_residues),
        "n_protein_ca": record.n_protein_ca,
        "nc_bytes": record.nc_bytes,
        "protein_motion_rmsd": record.protein_motion_rmsd,
        "ligand_relative_motion": record.ligand_relative_motion,
    }


def _features(records: list[TrajectoryRecord]) -> np.ndarray:
    raw = np.asarray([
        [
            np.log1p(r.n_frames),
            np.log1p(r.n_protein_atoms),
            np.log1p(r.n_ligand_atoms),
            np.log1p(r.nc_bytes),
            r.protein_motion_rmsd,
            r.ligand_relative_motion,
            r.n_ligand_atoms / max(r.n_protein_atoms, 1),
        ]
        for r in records
    ], dtype=np.float64)
    center = np.median(raw, axis=0)
    scale = np.median(np.abs(raw - center), axis=0)
    scale[scale < 1e-8] = 1.0
    return (raw - center) / scale


def _farthest_point_indices(features: np.ndarray, count: int) -> list[int]:
    if count >= len(features):
        return list(range(len(features)))
    center = features.mean(axis=0)
    selected = [int(np.argmin(np.sum((features - center) ** 2, axis=1)))]
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)
    min_dist[selected[0]] = -1.0
    while len(selected) < count:
        next_index = int(np.argmax(min_dist))
        selected.append(next_index)
        distance = np.linalg.norm(features - features[next_index], axis=1)
        min_dist = np.minimum(min_dist, distance)
        min_dist[selected] = -1.0
    return selected


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assign_splits(rows: list[dict]) -> None:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    split_by_index = {}
    n_train = int(round(len(rows) * 0.8))
    n_valid = int(round(len(rows) * 0.1))
    for rank, index in enumerate(order):
        split_by_index[int(index)] = (
            "train" if rank < n_train else "valid" if rank < n_train + n_valid else "test"
        )
    for index, row in enumerate(rows):
        row["split"] = split_by_index[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-root", type=Path, default=Path("/data/Model_coding/dynamic_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()

    records = trajectory_records(args.dynamic_root)
    if len(records) < args.count:
        raise RuntimeError(f"Only {len(records)} usable trajectories; need {args.count}.")
    features = _features(records)
    selected_indices = _farthest_point_indices(features, args.count)
    selected = [records[i] for i in selected_indices]
    rows = [_record_row(record) for record in selected]
    _assign_splits(rows)
    for rank, row in enumerate(rows):
        row["selection_rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "trajectory_audit_all.csv"
    selected_path = args.output_dir / "selected_trajectories_300.csv"
    _write_csv(audit_path, [_record_row(record) for record in records])
    _write_csv(selected_path, rows)

    manifest = {
        "dynamic_root": str(args.dynamic_root),
        "seed": SEED,
        "selection_method": "robust-scaled farthest-point sampling",
        "usable_trajectory_count": len(records),
        "selected_trajectory_count": len(selected),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "valid", "test")
        },
        "features": [
            "log_frame_count",
            "log_protein_atom_count",
            "log_ligand_atom_count",
            "log_trajectory_file_size",
            "protein_ca_motion_rmsd",
            "ligand_relative_centroid_motion",
            "ligand_to_protein_atom_ratio",
        ],
        "selected_trajectory_csv": str(selected_path),
        "audit_csv": str(audit_path),
    }
    (args.output_dir / "trajectory_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
