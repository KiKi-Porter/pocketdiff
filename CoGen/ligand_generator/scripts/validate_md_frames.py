from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=Path("data/md300/selected_frames_3000.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/md300/frame_reports/summary.json"),
    )
    args = parser.parse_args()

    rows = _read_rows(args.frames_csv)
    if not rows:
        raise ValueError(f"No frame rows found in {args.frames_csv}")

    split_by_trajectory: dict[str, str] = {}
    counts_by_split: dict[str, int] = {}
    missing = []
    invalid = []
    count_mismatch = []
    for row in rows:
        trajectory_id = row["trajectory_id"]
        previous_split = split_by_trajectory.setdefault(trajectory_id, row["split"])
        if previous_split != row["split"]:
            invalid.append(
                f"trajectory {trajectory_id} appears in multiple splits: "
                f"{previous_split}, {row['split']}"
            )
        counts_by_split[row["split"]] = counts_by_split.get(row["split"], 0) + 1
        sample_path = Path(row["sample_path"])
        if not sample_path.is_file():
            missing.append(str(sample_path))
            continue
        with np.load(sample_path) as sample:
            protein_pos = sample["protein_pos"]
            ligand_pos = sample["ligand_pos"]
            ca_mask = sample["protein_ca_mask"]
            if (
                protein_pos.ndim != 2
                or protein_pos.shape[-1] != 3
                or ligand_pos.ndim != 2
                or ligand_pos.shape[-1] != 3
                or ca_mask.shape != (protein_pos.shape[0],)
            ):
                invalid.append(f"invalid shapes in {sample_path}")
            if not np.isfinite(protein_pos).all() or not np.isfinite(ligand_pos).all():
                invalid.append(f"non-finite coordinates in {sample_path}")
            if protein_pos.shape[0] != int(row["pocket_atom_count"]):
                count_mismatch.append(str(sample_path))
            if ligand_pos.shape[0] != int(row["ligand_atom_count"]):
                count_mismatch.append(str(sample_path))
        if row["frame_index_convention"] != "cluster_1_based_equals_netcdf_index":
            invalid.append(f"unexpected frame convention in {sample_path}")

    summary = json.loads(args.report.read_text()) if args.report.is_file() else {}
    result = {
        "passed": not missing and not invalid and not count_mismatch,
        "row_count": len(rows),
        "trajectory_count": len(split_by_trajectory),
        "counts_by_split": counts_by_split,
        "missing_sample_count": len(missing),
        "invalid_sample_count": len(invalid),
        "count_mismatch_count": len(count_mismatch),
        "selection_summary": summary,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
