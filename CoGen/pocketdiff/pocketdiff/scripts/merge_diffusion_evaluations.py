"""Merge sharded PocketDiff diffusion evaluation reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from pocketdiff.training.diffusion import summarize_diffusion_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = [
        json.loads(path.read_text())
        for path in args.input
    ]
    if not reports:
        raise ValueError("at least one report is required")
    first = reports[0]
    for report in reports[1:]:
        for key in ("checkpoint", "prepared_cache", "steps", "prediction_type"):
            if report.get(key) != first.get(key):
                raise ValueError(f"shard mismatch for {key!r}")

    merged: Dict[str, object] = {
        "checkpoint": first.get("checkpoint"),
        "source_cache": first.get("source_cache"),
        "prepared_cache": first.get("prepared_cache"),
        "prediction_type": first.get("prediction_type"),
        "steps": first.get("steps"),
        "batch_size": first.get("batch_size"),
        "shard_count": len(reports),
        "valid": {},
        "test": {},
    }
    for split in ("valid", "test"):
        for step in first["steps"]:
            step_key = str(int(step))
            rows: List[Dict[str, object]] = []
            for report in reports:
                rows.extend(report.get(split, {}).get(step_key, {}).get("rows", []))
            if not rows:
                raise ValueError(f"no rows found for {split}/{step_key}")
            reference = next(
                report[split][step_key]
                for report in reports
                if report.get(split, {}).get(step_key)
            )
            merged[split][step_key] = summarize_diffusion_rows(
                rows,
                step_count=int(step),
                remaining_step_offset=float(
                    reference.get("remaining_step_offset", 0.0)
                ),
                motion_parameterization=str(
                    reference.get("motion_parameterization", "remaining")
                ),
                prediction_type=str(
                    reference.get("prediction_type", first["prediction_type"])
                ),
            )
            merged[split][step_key]["shard_rows"] = [
                len(report.get(split, {}).get(step_key, {}).get("rows", []))
                for report in reports
            ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    summary = {
        split: {
            step: {
                "apo_backbone": value["mean_apo_backbone_rmsd"],
                "final_backbone": value["mean_final_backbone_rmsd"],
                "apo_all_atom": value["mean_apo_rmsd"],
                "final_all_atom": value["mean_final_holo_rmsd"],
                "rigid_oracle_backbone": value[
                    "mean_rigid_oracle_backbone_rmsd"
                ],
            }
            for step, value in merged[split].items()
        }
        for split in ("valid", "test")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
