from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from datasets.md_utils import TopologyInfo, load_topology


POCKET_CUTOFF = 12.0
SELECTION_PROTOCOL = "existing_cluster_representatives_v1"
warnings.filterwarnings(
    "ignore",
    message="Cannot close a netcdf_file opened with mmap=True",
    category=RuntimeWarning,
)


def _heavy_atom_indices(topology: TopologyInfo, start: int, end: int) -> np.ndarray:
    return np.asarray(
        [
            index
            for index in range(start, end)
            if topology.atom_names[index].strip()[:1].upper() != "H"
        ],
        dtype=np.int64,
    )


def _read_cluster_representatives(
    directory: Path,
    n_frames: int,
) -> tuple[list[dict], dict]:
    info_path = directory / "cluster_info.txt"
    summary_path = directory / "cluster_summary.txt"
    if not info_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing cluster_info.txt or cluster_summary.txt in {directory}"
        )

    info_text = info_path.read_text(errors="replace")
    match = re.search(
        r"^#Representative frames:\s*(?P<frames>.+?)\s*$",
        info_text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Missing '#Representative frames' in {info_path}")
    representative_frames = [int(value) for value in match.group("frames").split()]
    if not representative_frames:
        raise ValueError(f"No representative frames in {info_path}")

    cluster_match = re.search(
        r"^#Clustering:\s*(?P<clusters>\d+)\s+clusters\s+"
        r"(?P<frames>\d+)\s+frames\s*$",
        info_text,
        flags=re.MULTILINE,
    )
    declared_frame_count = int(cluster_match.group("frames")) if cluster_match else None
    declared_cluster_count = int(cluster_match.group("clusters")) if cluster_match else None

    summary_lines = summary_path.read_text(errors="replace").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(summary_lines)
            if line.lstrip().startswith("#Cluster")
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"Missing cluster table header in {summary_path}")
    header = summary_lines[header_index].lstrip("#").split()
    if "Centroid" not in header:
        raise ValueError(f"Missing Centroid column in {summary_path}")
    centroid_index = header.index("Centroid")
    summary_centroids: dict[int, int] = {}
    for line in summary_lines[header_index + 1 :]:
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if len(tokens) <= centroid_index:
            continue
        try:
            cluster_id = int(tokens[0])
            centroid = int(tokens[centroid_index])
        except ValueError:
            continue
        summary_centroids[cluster_id] = centroid

    if len(summary_centroids) != len(representative_frames):
        raise ValueError(
            f"Representative count {len(representative_frames)} does not match "
            f"summary centroid count {len(summary_centroids)} in {directory}"
        )
    expected_cluster_ids = list(range(len(representative_frames)))
    if sorted(summary_centroids) != expected_cluster_ids:
        raise ValueError(
            f"Cluster IDs in {summary_path} are {sorted(summary_centroids)}, "
            f"expected {expected_cluster_ids}"
        )

    for cluster_id, representative in enumerate(representative_frames):
        centroid = summary_centroids[cluster_id]
        if representative != centroid:
            raise ValueError(
                f"Representative mismatch in {directory}: cluster {cluster_id} "
                f"has info={representative}, summary={centroid}"
            )

    if declared_cluster_count is not None and declared_cluster_count != len(
        representative_frames
    ):
        raise ValueError(
            f"Declared cluster count {declared_cluster_count} does not match "
            f"{len(representative_frames)} representatives in {directory}"
        )
    if declared_frame_count is not None and declared_frame_count >= n_frames:
        raise ValueError(
            f"Declared cluster frame count {declared_frame_count} is incompatible "
            f"with NetCDF frame count {n_frames} in {directory}"
        )

    if min(representative_frames) < 1:
        raise ValueError(
            f"Expected 1-based production frame numbers, got {representative_frames}"
        )
    if max(representative_frames) >= n_frames:
        raise ValueError(
            f"Representative frame {max(representative_frames)} is outside "
            f"NetCDF indices [0, {n_frames - 1}] in {directory}"
        )

    # The MD files contain an initial frame at index 0 followed by the
    # production frames numbered 1..N in the cluster files.
    rows = [
        {
            "cluster_id": cluster_id,
            "cluster_frame": representative,
            "netcdf_index": representative,
            "frame_index_convention": "cluster_1_based_equals_netcdf_index",
        }
        for cluster_id, representative in enumerate(representative_frames)
    ]
    metadata = {
        "cluster_info": str(info_path),
        "cluster_summary": str(summary_path),
        "declared_cluster_count": declared_cluster_count,
        "declared_cluster_frame_count": declared_frame_count,
        "representative_frame_count": len(rows),
        "frame_index_convention": "cluster_1_based_equals_netcdf_index",
    }
    return rows, metadata


def _reimage_ligand_to_protein(
    frame: np.ndarray,
    topology: TopologyInfo,
    box: np.ndarray | None,
) -> np.ndarray:
    if (
        box is None
        or topology.protein_atom_indices.size == 0
        or topology.ligand_atom_indices.size == 0
    ):
        return frame
    output = frame.copy()
    protein_center = output[topology.protein_atom_indices].mean(axis=0)
    ligand_center = output[topology.ligand_atom_indices].mean(axis=0)
    safe_box = np.where(box > 1e-6, box, 1.0)
    shift = np.round((ligand_center - protein_center) / safe_box) * box
    output[topology.ligand_atom_indices] -= shift
    return output


def _pocket_residue_indices(
    topology: TopologyInfo,
    frame: np.ndarray,
    box: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    ligand = frame[topology.ligand_atom_indices]
    selected_ranges: list[tuple[int, int]] = []
    selected_residues: list[dict] = []
    for residue_index, (start, end, residue_name) in enumerate(
        topology.protein_residue_ranges
    ):
        heavy = _heavy_atom_indices(topology, start, end)
        if heavy.size == 0:
            continue
        distances = frame[heavy, None, :] - ligand[None, :, :]
        if box is not None:
            safe_box = np.where(box > 1e-6, box, 1.0)
            distances = distances - np.round(
                distances / safe_box[None, None, :]
            ) * safe_box[None, None, :]
        min_distance = float(np.sqrt(np.sum(distances * distances, axis=-1)).min())
        if min_distance <= POCKET_CUTOFF:
            selected_ranges.append((start, end))
            selected_residues.append(
                {
                    "residue_index": residue_index,
                    "residue_name": residue_name,
                    "start_atom": start,
                    "end_atom": end,
                    "min_heavy_atom_ligand_distance": min_distance,
                }
            )
    if not selected_ranges:
        raise ValueError("No protein residue within pocket cutoff.")

    atom_indices = np.asarray(
        [
            index
            for start, end in selected_ranges
            for index in range(start, end)
        ],
        dtype=np.int64,
    )
    ca_indices = np.asarray(
        [
            index
            for start, end in selected_ranges
            for index in range(start, end)
            if topology.atom_names[index].strip() == "CA"
        ],
        dtype=np.int64,
    )
    return atom_indices, ca_indices, selected_residues


def _safe_sample_name(trajectory_id: str, cluster_id: int, frame: int) -> str:
    clean_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", trajectory_id)
    return f"{clean_id}__cluster_{cluster_id:02d}__frame_{frame:04d}.npz"


def _write_cropped_sample(
    output_path: Path,
    frame: np.ndarray,
    topology: TopologyInfo,
    pocket_atom_indices: np.ndarray,
    pocket_ca_indices: np.ndarray,
    selected_residues: list[dict],
    box: np.ndarray | None,
) -> None:
    residue_index_by_atom = np.full(len(topology.atom_names), -1, dtype=np.int64)
    residue_name_by_atom = np.full(len(topology.atom_names), "", dtype="<U8")
    for residue_index, (start, end, residue_name) in enumerate(
        topology.protein_residue_ranges
    ):
        residue_index_by_atom[start:end] = residue_index
        residue_name_by_atom[start:end] = residue_name
    ligand_indices = topology.ligand_atom_indices
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        protein_pos=frame[pocket_atom_indices].astype(np.float32, copy=False),
        ligand_pos=frame[ligand_indices].astype(np.float32, copy=False),
        protein_global_atom_index=pocket_atom_indices,
        ligand_global_atom_index=ligand_indices,
        protein_atom_name=np.asarray(
            [topology.atom_names[index].strip() for index in pocket_atom_indices],
            dtype="<U8",
        ),
        ligand_atom_name=np.asarray(
            [topology.atom_names[index].strip() for index in ligand_indices],
            dtype="<U8",
        ),
        protein_residue_index=residue_index_by_atom[pocket_atom_indices],
        protein_residue_name=residue_name_by_atom[pocket_atom_indices],
        protein_ca_mask=np.isin(pocket_atom_indices, pocket_ca_indices),
        ligand_residue_name=np.asarray(topology.ligand_residue_names, dtype="<U8"),
        box=(
            np.asarray(box, dtype=np.float32)
            if box is not None
            else np.zeros(3, dtype=np.float32)
        ),
        selected_residues=np.asarray(
            json.dumps(selected_residues),
            dtype="<U4096",
        ),
    )


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _process_trajectory(
    row: dict,
    sample_dir: Path,
    report_dir: Path,
    cluster_count: int,
) -> tuple[list[dict], dict]:
    directory = Path(row["directory"])
    topology = load_topology(
        row["topology_path"],
        directory / "ligand.prmtop",
    )
    trajectory_path = row["trajectory_path"]
    output_rows: list[dict] = []
    sample_reports: list[dict] = []
    skipped_reports: list[dict] = []

    with netcdf_file(trajectory_path, mmap=True) as nc:
        coordinates = nc.variables["coordinates"]
        cell_lengths_var = nc.variables.get("cell_lengths")
        n_frames = int(coordinates.shape[0])
        representatives, cluster_metadata = _read_cluster_representatives(
            directory,
            n_frames,
        )
        if len(representatives) != cluster_count:
            raise ValueError(
                f"{directory}: expected {cluster_count} representative frames, "
                f"found {len(representatives)}"
            )

        for representative in representatives:
            netcdf_index = representative["netcdf_index"]
            frame = np.array(
                coordinates[netcdf_index],
                dtype=np.float32,
                copy=True,
            )
            box = (
                np.array(
                    cell_lengths_var[netcdf_index],
                    dtype=np.float32,
                    copy=True,
                )
                if cell_lengths_var is not None
                else None
            )
            frame = _reimage_ligand_to_protein(frame, topology, box)
            try:
                pocket_atom_indices, pocket_ca_indices, selected_residues = (
                    _pocket_residue_indices(topology, frame, box)
                )
            except ValueError as exc:
                skipped_reports.append(
                    {
                        "cluster_id": representative["cluster_id"],
                        "cluster_frame": representative["cluster_frame"],
                        "netcdf_index": netcdf_index,
                        "reason": str(exc),
                    }
                )
                continue
            sample_name = _safe_sample_name(
                row["trajectory_id"],
                representative["cluster_id"],
                representative["cluster_frame"],
            )
            sample_path = sample_dir / sample_name
            _write_cropped_sample(
                output_path=sample_path,
                frame=frame,
                topology=topology,
                pocket_atom_indices=pocket_atom_indices,
                pocket_ca_indices=pocket_ca_indices,
                selected_residues=selected_residues,
                box=box,
            )
            output_row = {
                "trajectory_id": row["trajectory_id"],
                "pdb_id": row["pdb_id"],
                "split": row["split"],
                "directory": row["directory"],
                "trajectory_path": row["trajectory_path"],
                "topology_path": row["topology_path"],
                "reference_path": row["reference_path"],
                "cluster_id": representative["cluster_id"],
                "cluster_frame": representative["cluster_frame"],
                "netcdf_index": netcdf_index,
                "frame_index": netcdf_index,
                "frame_index_convention": representative["frame_index_convention"],
                "sample_path": str(sample_path),
                "pocket_atom_count": int(pocket_atom_indices.size),
                "pocket_ca_count": int(pocket_ca_indices.size),
                "pocket_residue_count": len(selected_residues),
                "ligand_atom_count": int(topology.ligand_atom_indices.size),
            }
            output_rows.append(output_row)
            sample_reports.append(
                {
                    **output_row,
                    "pocket_global_atom_indices": pocket_atom_indices.tolist(),
                    "ligand_global_atom_indices": topology.ligand_atom_indices.tolist(),
                    "selected_residues": selected_residues,
                    "box": box.tolist() if box is not None else None,
                }
            )

    trajectory_report = {
        "trajectory_id": row["trajectory_id"],
        "directory": row["directory"],
        "n_frames": n_frames,
        "representative_frames": cluster_metadata,
        "selected_frame_count": len(output_rows),
        "skipped_frame_count": len(skipped_reports),
        "skipped_frames": skipped_reports,
        "samples": sample_reports,
    }
    report_name = _safe_sample_name(row["trajectory_id"], 0, 0).split("__")[0]
    report_path = report_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(trajectory_report, indent=2) + "\n")
    return output_rows, trajectory_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing cluster representatives, crop one independent "
            "TargetDiff pocket sample per representative frame."
        )
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=Path("data/md300/selected_trajectories_300.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/md300/selected_frames_3000.csv"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/md300/frame_reports"),
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("data/md300/frame_samples"),
    )
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--trajectory-limit", type=int, default=None)
    args = parser.parse_args()

    rows = _read_rows(args.trajectory_csv)
    if args.trajectory_limit is not None:
        rows = rows[: args.trajectory_limit]
    if not rows:
        raise ValueError(f"No trajectories found in {args.trajectory_csv}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.sample_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict] = []
    reports: dict[str, dict] = {}
    for index, row in enumerate(rows, start=1):
        selected, report = _process_trajectory(
            row=row,
            sample_dir=args.sample_dir,
            report_dir=args.report_dir,
            cluster_count=args.clusters,
        )
        output_rows.extend(selected)
        reports[row["trajectory_id"]] = report
        if index % 10 == 0 or index == len(rows):
            print(
                f"processed {index}/{len(rows)} trajectories; "
                f"samples={len(output_rows)}",
                flush=True,
            )

    fields = list(output_rows[0])
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    reports_path = args.report_dir / "reports.json"
    reports_path.write_text(json.dumps(reports, indent=2) + "\n")
    split_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    for item in output_rows:
        split_counts[item["split"]] = split_counts.get(item["split"], 0) + 1
    for report in reports.values():
        split = next(
            item["split"]
            for item in rows
            if item["trajectory_id"] == report["trajectory_id"]
        )
        skipped_counts[split] = skipped_counts.get(split, 0) + int(
            report["skipped_frame_count"]
        )
    summary = {
        "selection_protocol": SELECTION_PROTOCOL,
        "trajectory_csv": str(args.trajectory_csv),
        "trajectory_count": len(rows),
        "cluster_count": args.clusters,
        "selected_frame_count": len(output_rows),
        "expected_max_frame_count": len(rows) * args.clusters,
        "split_sample_counts": split_counts,
        "skipped_frame_count": sum(
            int(report["skipped_frame_count"]) for report in reports.values()
        ),
        "split_skipped_frame_counts": skipped_counts,
        "trajectories_with_fewer_than_cluster_count_samples": sum(
            int(report["selected_frame_count"] < args.clusters)
            for report in reports.values()
        ),
        "trajectories_with_zero_valid_samples": sum(
            int(report["selected_frame_count"] == 0)
            for report in reports.values()
        ),
        "pocket_cutoff_angstrom": POCKET_CUTOFF,
        "frame_index_convention": "cluster_1_based_equals_netcdf_index",
        "sample_dir": str(args.sample_dir),
        "output_csv": str(args.output_csv),
        "reports_path": str(reports_path),
    }
    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest_path = args.output_csv.parent / "frame_selection_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
