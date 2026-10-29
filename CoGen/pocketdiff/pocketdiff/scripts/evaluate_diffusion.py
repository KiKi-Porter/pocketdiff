"""Evaluate a PocketDiff velocity checkpoint with fixed 20/50/100-step protocols."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffTrainingLigandConditionProvider,
)
from pocketdiff.training.diffusion import (
    DiffusionTrainConfig,
    evaluate_diffusion_sampler,
    load_apo2mol_slices,
    load_diffusion_checkpoint,
    summarize_diffusion_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    cache_group = parser.add_mutually_exclusive_group(required=True)
    cache_group.add_argument("--source-cache", type=Path)
    cache_group.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--holdout-seed", type=int, default=5701)
    parser.add_argument("--steps", default="20,50,100")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of disjoint complexes evaluated in one model forward batch.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--targetdiff-checkpoint", type=Path, default=None)
    parser.add_argument("--targetdiff-root", type=Path, default=None)
    parser.add_argument("--ligand-noise-t-min", type=int, default=0)
    parser.add_argument("--ligand-noise-t-max", type=int, default=199)
    parser.add_argument("--ligand-clean-probability", type=float, default=0.20)
    parser.add_argument("--ligand-condition-seed", type=int, default=61021)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.targetdiff_root is not None:
        os.environ["POCKETDIFF_TARGETDIFF_ROOT"] = str(
            Path(args.targetdiff_root).resolve()
        )
    loaded = load_diffusion_checkpoint(args.checkpoint, map_location=args.device)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must lie in [0, shard-count)")
    if loaded.model.encoder_backend == "targetdiff" and args.batch_size != 1:
        raise ValueError(
            "TargetDiff evaluation requires --batch-size 1: the official "
            "GPU KNN graph is not numerically equivalent for multi-graph batches."
        )
    training_config = loaded.payload.get("training_config", {})
    config = DiffusionTrainConfig(
        source_cache=str(args.source_cache) if args.source_cache else None,
        prepared_cache=str(args.prepared_cache) if args.prepared_cache else None,
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
        architecture_variant=str(
            loaded.payload.get("model_config", {}).get(
                "architecture_variant", "v45"
            )
        ),
        target_parameterization=str(
            training_config.get(
                "target_parameterization",
                loaded.payload.get("model_config", {}).get(
                    "target_parameterization",
                    loaded.model.prediction_type,
                ),
            )
        ),
        target_steps=int(
            training_config.get(
                "target_steps",
                loaded.payload.get("model_config", {}).get("target_steps", 20),
            )
        ),
    )
    slices = load_apo2mol_slices(config)
    steps = tuple(int(value) for value in args.steps.split(",") if value.strip())
    ligand_conditioner = None
    if args.targetdiff_checkpoint is not None:
        adapter = TargetDiffAdapter.from_checkpoint(
            args.targetdiff_checkpoint,
            targetdiff_root=args.targetdiff_root,
            device=args.device,
        )
        ligand_conditioner = TargetDiffTrainingLigandConditionProvider(
            adapter,
            t_min=args.ligand_noise_t_min,
            t_max=args.ligand_noise_t_max,
            clean_probability=args.ligand_clean_probability,
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "source_cache": str(args.source_cache) if args.source_cache else None,
        "prepared_cache": str(args.prepared_cache) if args.prepared_cache else None,
        "prediction_type": loaded.model.prediction_type,
        "steps": list(steps),
        "batch_size": int(args.batch_size),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "ligand_condition": (
            {
                "provider": "TargetDiffTrainingLigandConditionProvider",
                "targetdiff_checkpoint": str(args.targetdiff_checkpoint),
                "t_min": args.ligand_noise_t_min,
                "t_max": args.ligand_noise_t_max,
                "clean_probability": args.ligand_clean_probability,
                "seed": args.ligand_condition_seed,
            }
            if ligand_conditioner is not None
            else {"provider": "clean Apo2Mol reference"}
        ),
        "valid": {},
        "test": {},
    }

    def write_report() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for split in ("valid", "test"):
        values = slices[split].values[
            args.shard_index :: args.shard_count
        ]
        for step_count in steps:
            print(
                f"[evaluate] start split={split} steps={step_count} "
                f"samples={len(values)} batch_size={args.batch_size}",
                flush=True,
            )
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
                ligand_conditioner=ligand_conditioner,
                ligand_condition_seed=args.ligand_condition_seed,
                batch_size=args.batch_size,
                progress_label=f"{split}/{step_count}",
            )
            write_report()
            print(
                f"[evaluate] completed split={split} steps={step_count}",
                flush=True,
            )
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
