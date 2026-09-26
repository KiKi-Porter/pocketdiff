"""Train the independent PocketDiff diffusion core from Apo2Mol records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocketdiff.training.diffusion import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SPLIT_PATH,
    DiffusionTrainConfig,
    run_diffusion_training,
)


def _parse_float_tuple(value: str):
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_int_tuple(value: str):
    return tuple(int(item) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--source-cache", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--holdout-split", default="valid")
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--valid-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--holdout-count", type=int, default=16)
    parser.add_argument("--train-seed", type=int, default=5700)
    parser.add_argument("--holdout-seed", type=int, default=5701)
    parser.add_argument("--diffusion-seed", type=int, default=5700)
    parser.add_argument("--updates", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--self-state-probability", type=float, default=0.0)
    parser.add_argument("--self-state-steps", type=int, default=1)
    parser.add_argument("--self-state-rollout-steps", type=int, default=8)
    parser.add_argument("--self-state-warmup", type=int, default=0)
    parser.add_argument("--cache-manifest", type=Path, default=None)
    parser.add_argument("--no-verify-cache-sources", action="store_true")
    parser.add_argument("--times", default="0.05,0.25,0.5,0.75,0.95,1.0")
    parser.add_argument("--time-min", type=float, default=0.02)
    parser.add_argument("--time-max", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--encoder-backend",
        choices=("scalar", "targetdiff", "dynamicbind"),
        default="scalar",
    )
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--score-normalize", action="store_true")
    parser.add_argument("--score-floor", type=float, default=0.1)
    parser.add_argument("--motion-parameterization", choices=("remaining", "bridge_rate"), default="remaining")
    parser.add_argument("--sampler-steps", default="4,8")
    parser.add_argument("--remaining-step-offset", type=float, default=0.0)
    parser.add_argument("--translation-noise-scale", type=float, default=0.10)
    parser.add_argument("--rotation-noise-scale", type=float, default=0.05)
    parser.add_argument("--chi-noise-scale", type=float, default=0.05)
    parser.add_argument(
        "--prediction-type", choices=("remaining", "velocity"), default="velocity"
    )
    parser.add_argument("--backbone-endpoint-weight", type=float, default=2.0)
    parser.add_argument("--continuity-weight", type=float, default=0.10)
    parser.add_argument("--direction-weight", type=float, default=0.05)
    parser.add_argument("--direction-threshold", type=float, default=0.05)
    parser.add_argument("--motion-bucket-count", type=int, default=4)
    parser.add_argument(
        "--no-motion-bucket-balance",
        action="store_true",
        help="Sample the training list sequentially instead of balancing motion buckets",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DiffusionTrainConfig(
        data_root=str(args.data_root),
        split_path=str(args.split_path),
        source_cache=str(args.source_cache) if args.source_cache else None,
        device=args.device,
        train_split=args.train_split,
        valid_split=args.valid_split,
        test_split=args.test_split,
        holdout_split=args.holdout_split,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        holdout_count=args.holdout_count,
        train_seed=args.train_seed,
        holdout_seed=args.holdout_seed,
        diffusion_seed=args.diffusion_seed,
        updates=args.updates,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        self_state_probability=args.self_state_probability,
        self_state_steps=args.self_state_steps,
        self_state_rollout_steps=args.self_state_rollout_steps,
        self_state_warmup=args.self_state_warmup,
        cache_manifest=str(args.cache_manifest) if args.cache_manifest else None,
        verify_cache_sources=not args.no_verify_cache_sources,
        times=_parse_float_tuple(args.times),
        time_min=args.time_min,
        time_max=args.time_max,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        encoder_backend=args.encoder_backend,
        endpoint_weight=args.endpoint_weight,
        score_normalize=args.score_normalize,
        score_floor=args.score_floor,
        motion_parameterization=args.motion_parameterization,
        sampler_steps=_parse_int_tuple(args.sampler_steps),
        remaining_step_offset=args.remaining_step_offset,
        translation_noise_scale=args.translation_noise_scale,
        rotation_noise_scale=args.rotation_noise_scale,
        chi_noise_scale=args.chi_noise_scale,
        prediction_type=args.prediction_type,
        backbone_endpoint_weight=args.backbone_endpoint_weight,
        continuity_weight=args.continuity_weight,
        direction_weight=args.direction_weight,
        direction_threshold=args.direction_threshold,
        motion_bucket_count=args.motion_bucket_count,
        motion_bucket_balance=not args.no_motion_bucket_balance,
    )
    report = run_diffusion_training(config, args.output_dir)
    print(json.dumps({
        "passed": report["passed"],
        "learning_goal_met": report["learning_goal_met"],
        "checkpoint": report["checkpoint"],
        "train_final": report["train"]["mean_final_holo_rmsd"],
        "train_apo": report["train"]["mean_apo_rmsd"],
        "valid_final": report["valid"]["mean_final_holo_rmsd"],
        "valid_apo": report["valid"]["mean_apo_rmsd"],
        "test_final": report["test"]["mean_final_holo_rmsd"],
        "test_apo": report["test"]["mean_apo_rmsd"],
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
