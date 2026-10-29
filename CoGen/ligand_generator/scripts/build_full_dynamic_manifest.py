from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def _split_for_trajectory(
    trajectory_id: str,
    valid_fraction: float,
    test_fraction: float,
) -> str:
    value = int(hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()[:8], 16)
    bucket = value / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + valid_fraction:
        return "valid"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign deterministic trajectory-level splits to all MD trajectories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/md300/trajectory_audit_all.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/full_dynamic/selected_trajectories_all.csv"),
    )
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    args = parser.parse_args()

    if args.valid_fraction < 0 or args.test_fraction < 0:
        raise ValueError("Split fractions must be non-negative")
    if args.valid_fraction + args.test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No trajectory rows found in {args.input}")

    fields = list(rows[0]) + ["split", "selection_rank"]
    counts = {"train": 0, "valid": 0, "test": 0}
    output_rows = []
    for row in rows:
        split = _split_for_trajectory(
            row["trajectory_id"],
            valid_fraction=args.valid_fraction,
            test_fraction=args.test_fraction,
        )
        output = dict(row)
        output["split"] = split
        output["selection_rank"] = ""
        output_rows.append(output)
        counts[split] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        {
            "input": str(args.input),
            "output": str(args.output),
            "trajectory_count": len(output_rows),
            "split_counts": counts,
            "frames_per_trajectory": 10,
            "split_protocol": "sha256_trajectory_id_v1",
        }
    )


if __name__ == "__main__":
    main()
