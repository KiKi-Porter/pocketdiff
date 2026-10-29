"""Run deterministic apo-start PocketDiff diffusion sampling from a checkpoint."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.training.diffusion import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SPLIT_PATH,
    coordinate_rmsd,
    sample_checkpoint_on_complex,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--remaining-step-offset", type=float, default=0.0)
    parser.add_argument("--motion-parameterization", choices=("remaining", "bridge_rate"), default=None)
    parser.add_argument("--prediction-type", choices=("remaining", "velocity"), default=None)
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional .pt file for all coordinate states")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with args.split_path.open("rb") as handle:
        split = pickle.load(handle)
    record = split[args.split][args.index]
    value = Apo2MolAdapter(args.data_root).convert_record(record)
    trajectory = sample_checkpoint_on_complex(
        args.checkpoint,
        value,
        steps=args.steps,
        remaining_step_offset=args.remaining_step_offset,
        motion_parameterization=args.motion_parameterization,
        prediction_type=args.prediction_type,
    )
    final = trajectory.states[-1]
    report = {
        "sample_id": value.sample_id,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "index": args.index,
        "steps": args.steps,
        "state_count": len(trajectory.states),
        "times": list(trajectory.times),
        "finite": all(bool(torch.isfinite(state).all()) for state in trajectory.states),
        "apo_holo_rmsd": coordinate_rmsd(value.protein_pos_apo, value.protein_pos_holo),
        "final_holo_rmsd": coordinate_rmsd(final, value.protein_pos_holo),
        "mean_displacement_from_apo": float(torch.linalg.vector_norm(final - trajectory.states[0], dim=-1).mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.trajectory is not None:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_id": value.sample_id, "states": trajectory.states, "times": trajectory.times}, args.trajectory)
    print(json.dumps(report, indent=2))
    if not report["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
