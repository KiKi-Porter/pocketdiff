"""Evaluate a PocketDiff velocity checkpoint with fixed 20/50/100-step protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocketdiff.training.diffusion import (
    DiffusionTrainConfig,
    evaluate_diffusion_sampler,
    load_apo2mol_slices,
    load_diffusion_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--holdout-seed", type=int, default=5701)
    parser.add_argument("--steps", default="20,50,100")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = load_diffusion_checkpoint(args.checkpoint, map_location=args.device)
    training_config = loaded.payload.get("training_config", {})
    config = DiffusionTrainConfig(
        source_cache=str(args.source_cache),
        device=args.device,
        train_count=1,
        valid_count=args.valid_count,
        test_count=args.test_count,
        holdout_count=args.valid_count,
        holdout_seed=args.holdout_seed,
        prediction_type=str(
            loaded.payload.get("model_config", {}).get(
                "prediction_type",
                training_config.get("prediction_type", "velocity"),
            )
        ),
    )
    slices = load_apo2mol_slices(config)
    steps = tuple(int(value) for value in args.steps.split(",") if value.strip())
    report = {
        "checkpoint": str(args.checkpoint),
        "source_cache": str(args.source_cache),
        "prediction_type": loaded.model.prediction_type,
        "steps": list(steps),
        "valid": {},
        "test": {},
    }
    for split in ("valid", "test"):
        values = slices[split].values
        for step_count in steps:
            report[split][str(step_count)] = evaluate_diffusion_sampler(
                loaded.model,
                values,
                steps=(step_count,),
                prediction_type=loaded.model.prediction_type,
                motion_parameterization=str(
                    training_config.get("motion_parameterization", "remaining")
                ),
                remaining_step_offset=float(
                    training_config.get("remaining_step_offset", 0.0)
                ),
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {}
    for split in ("valid", "test"):
        summary[split] = {
            step: {
                "apo_backbone": value["mean_apo_backbone_rmsd"],
                "final_backbone": value["mean_final_backbone_rmsd"],
                "apo_all_atom": value["mean_apo_rmsd"],
                "final_all_atom": value["mean_final_holo_rmsd"],
                "rigid_oracle_backbone": value[
                    "mean_rigid_oracle_backbone_rmsd"
                ],
            }
            for step, value in report[split].items()
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
