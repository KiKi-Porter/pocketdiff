"""Run the bounded Phase 70 protocol comparison without overwriting prior runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffLigandConditionProvider,
)
from pocketdiff.training.diffusion import DiffusionTrainConfig, run_diffusion_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--targetdiff-checkpoint",
        type=Path,
        default=Path("targetdiff-main/targetdiff-main/pretrained_models/pretrained_diffusion.pt"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--valid-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    args = parser.parse_args()
    common = dict(
        device=args.device,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        holdout_count=args.valid_count,
        train_seed=5700,
        holdout_seed=5701,
        diffusion_seed=5700,
        updates=args.updates,
        times=(0.05, 0.5, 0.95, 1.0),
        sampler_steps=(4, 8),
        encoder_backend="dynamicbind",
        endpoint_weight=1.0,
    )
    configs = {
        "bounded_baseline": DiffusionTrainConfig(
            **common,
            batch_size=1,
            gradient_accumulation_steps=1,
            self_state_probability=0.0,
        ),
        "bounded_anchored_exposure": DiffusionTrainConfig(
            **common,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            self_state_probability=0.5,
            self_state_steps=2,
            self_state_rollout_steps=8,
        ),
        "bounded_targetdiff_condition": DiffusionTrainConfig(
            **common,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            self_state_probability=0.5,
            self_state_steps=2,
            self_state_rollout_steps=8,
        ),
    }
    targetdiff_adapter = TargetDiffAdapter.from_checkpoint(
        args.targetdiff_checkpoint,
        targetdiff_root=Path("targetdiff-main/targetdiff-main"),
        device=args.device,
    )
    targetdiff_conditioner = TargetDiffLigandConditionProvider(targetdiff_adapter)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, config in configs.items():
        conditioner = (
            targetdiff_conditioner
            if name == "bounded_targetdiff_condition"
            else None
        )
        report = run_diffusion_training(
            config,
            args.output_root / name,
            ligand_conditioner=conditioner,
        )
        summary[name] = {
            "passed": report["passed"],
            "learning_goal_met": report["learning_goal_met"],
            "initial_loss": report["initial_loss"],
            "final_loss": report["final_loss"],
            "train": {
                "apo": report["train"]["mean_apo_rmsd"],
                "final": report["train"]["mean_final_holo_rmsd"],
            },
            "valid": {
                "apo": report["valid"]["mean_apo_rmsd"],
                "final": report["valid"]["mean_final_holo_rmsd"],
            },
            "test": {
                "apo": report["test"]["mean_apo_rmsd"],
                "final": report["test"]["mean_final_holo_rmsd"],
            },
        }
    output = args.output_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
