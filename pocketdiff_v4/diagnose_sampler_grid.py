from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .batching import collate_complexes
from .cache import load_cache
from .evaluate import (
    _batches,
    _move,
    _sample_metrics,
    _summarize,
    _trajectory_metrics,
)
from .model import PocketDiffV4Model
from .sampler import sample_complexes_with_trajectory


def _arguments():
    parser = argparse.ArgumentParser(
        description="Compare PocketDiff sampler schedules on one split."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=12000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--step-counts", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--motion-scales", nargs="+", type=float, default=None)
    parser.add_argument("--initial-noise-scale", type=float, default=0.0)
    parser.add_argument("--disable-chi", action="store_true")
    parser.add_argument("--save-rows", action="store_true")
    return parser.parse_args()


def _load_model(checkpoint_path: str, device: torch.device):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != "pocketdiff-v4-checkpoint":
        raise ValueError("not a PocketDiff v4 training checkpoint")
    config = checkpoint["config"]
    model = PocketDiffV4Model(
        hidden=int(config["hidden"]),
        vector_channels=int(config["vector_channels"]),
        layers=int(config["layers"]),
        radial_count=int(config.get("radial_count", 24)),
        radial_cutoff=float(config.get("radial_cutoff", 16.0)),
        max_translation=float(config.get("max_translation", 8.0)),
        max_rotation=float(config.get("max_rotation", torch.pi)),
        max_chi_step=float(config.get("max_chi_step", torch.pi)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return checkpoint, model


def _schedule_configs():
    return {
        "remaining": {
            "schedule_type": "remaining",
        },
        "capped_025": {
            "schedule_type": "clipped",
            "max_fraction": 0.25,
        },
        "capped_033": {
            "schedule_type": "clipped",
            "max_fraction": 0.33,
        },
        "damped_025": {
            "schedule_type": "damped",
            "damping": 0.25,
        },
        "fixed_020": {
            "schedule_type": "fixed",
            "fixed_fraction": 0.20,
        },
        "fixed_025": {
            "schedule_type": "fixed",
            "fixed_fraction": 0.25,
        },
    }


@torch.inference_mode()
def main():
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("sampler grid requires CUDA")
    step_counts = args.step_counts or [args.steps]
    motion_scales = args.motion_scales or [args.motion_scale]
    if (
        any(steps <= 0 for steps in step_counts)
        or any(not 0.0 <= scale <= 1.0 for scale in motion_scales)
        or args.batch_size <= 0
        or args.max_nodes <= 0
    ):
        raise ValueError("steps, batch-size, and max-nodes must be positive")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cache = load_cache(args.data)
    checkpoint, model = _load_model(args.checkpoint, device)
    samples = list(cache["splits"][args.split])
    if args.max_samples:
        samples = samples[: args.max_samples]

    results = {}
    for steps in step_counts:
        for motion_scale in motion_scales:
            for name, schedule in _schedule_configs().items():
                rows = []
                for chosen in _batches(samples, args.batch_size, args.max_nodes):
                    batch = _move(collate_complexes(chosen), device)
                    inputs = batch["input"]
                    sample_ids = batch["target"]["sample_id"]
                    predicted, trajectory = sample_complexes_with_trajectory(
                        model,
                        inputs,
                        sample_ids,
                        steps=steps,
                        seed=args.seed,
                        motion_scale=motion_scale,
                        initial_noise_scale=args.initial_noise_scale,
                        disable_chi=args.disable_chi,
                        **schedule,
                    )
                    atom_ptr = inputs["atom_ptr"].tolist()
                    for graph_id, record in enumerate(chosen):
                        start, end = atom_ptr[graph_id : graph_id + 2]
                        row = _sample_metrics(record, predicted[start:end])
                        row["trajectory"] = _trajectory_metrics(
                            record,
                            [state[start:end] for state in trajectory],
                            disable_chi=args.disable_chi,
                            **schedule,
                        )
                        rows.append(row)
                key = "k%d_m%.2f_%s" % (steps, motion_scale, name)
                results[key] = {
                    "steps": steps,
                    "motion_scale": motion_scale,
                    "schedule": schedule,
                    "summary": _summarize(rows),
                }
                if args.save_rows:
                    results[key]["rows"] = sorted(
                        rows, key=lambda row: row["sample_id"]
                    )

    selected_name = max(
        results,
        key=lambda name: (
            results[name]["summary"].get(
                "improvement_backbone_rmsd_mean", float("-inf")
            ),
            results[name]["summary"].get(
                "improvement_atom_rmsd_mean", float("-inf")
            ),
            -results[name]["summary"].get("sample_displacement_mean", float("inf")),
        ),
    )
    output = {
        "format": "pocketdiff-v4-sampler-grid",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_update": int(checkpoint["update"]),
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "sample_count": len(samples),
        "step_counts": step_counts,
        "seed": args.seed,
        "motion_scales": motion_scales,
        "initial_noise_scale": args.initial_noise_scale,
        "selection_metric": "improvement_backbone_rmsd_mean",
        "selected_schedule": selected_name,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            "steps": result["steps"],
            "motion_scale": result["motion_scale"],
            "schedule": result["schedule"],
            "improvement_backbone_rmsd_mean": (
                result["summary"].get("improvement_backbone_rmsd_mean")
            ),
            "improvement_atom_rmsd_mean": (
                result["summary"].get("improvement_atom_rmsd_mean")
            ),
            "sample_displacement_mean": (
                result["summary"].get("sample_displacement_mean")
            ),
        }
        for name, result in results.items()
    }, indent=2, sort_keys=True))
    selected = results[selected_name]
    print(json.dumps({
        "selected": selected_name,
        "steps": selected["steps"],
        "motion_scale": selected["motion_scale"],
        "schedule": selected["schedule"],
    }))


if __name__ == "__main__":
    main()
