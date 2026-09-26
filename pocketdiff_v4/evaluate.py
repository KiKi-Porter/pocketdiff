from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

from .batching import collate_complexes
from .cache import load_cache, source_fingerprint
from .constants import NUM_CHI
from .geometry import apply_motion, residue_frames
from .sampler import (
    sample_complexes,
    sample_complexes_with_trajectory,
    schedule_fraction,
)
from pocketdiff.geometry.bridge import remaining_transform_current_to_holo


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="pocketdiff_v4/data/residue_graphs.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=12000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--initial-noise-scale", type=float, default=0.0)
    parser.add_argument(
        "--schedule-type",
        choices=("remaining", "clipped", "damped", "fixed"),
        default=None,
    )
    parser.add_argument("--max-fraction", type=float, default=None)
    parser.add_argument("--fixed-fraction", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--disable-chi", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=("train", "valid", "test"),
    )
    parser.add_argument("--max-per-split", type=int, default=0)
    return parser.parse_args()


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _coord_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt((left - right).square().mean()).item())


def _atom_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.sqrt((left - right).square().sum(dim=-1).mean()).item()
    )


def _masked_atom_rmsd(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor
) -> float:
    if not bool(mask.any()):
        return 0.0
    return _atom_rmsd(left[mask], right[mask])


def _direction_cosine(apo: torch.Tensor, holo: torch.Tensor, predicted: torch.Tensor) -> float:
    target = holo - apo
    motion = predicted - apo
    numerator = (target * motion).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(target, dim=-1)
        * torch.linalg.vector_norm(motion, dim=-1)
    ).clamp_min(1e-8)
    # Tiny apo/holo differences have numerically arbitrary directions.
    valid = torch.linalg.vector_norm(target, dim=-1) > 0.05
    if not bool(valid.any()):
        return 0.0
    return float((numerator[valid] / denominator[valid]).mean().item())


def _sample_metrics(record: Dict[str, object], predicted: torch.Tensor):
    inputs, targets = record["input"], record["target"]
    apo = inputs["apo_pos"].to(predicted.device)
    holo = targets["holo_pos"].to(predicted.device)
    ligand = inputs["ligand_pos"].to(predicted.device)
    pocket = torch.cdist(apo, ligand).min(dim=-1).values <= 8.0
    if not bool(pocket.any()):
        pocket = torch.ones(apo.shape[0], dtype=torch.bool, device=apo.device)
    frame_index = inputs["frame_index"].to(predicted.device)
    ca_mask = torch.zeros(apo.shape[0], dtype=torch.bool, device=apo.device)
    ca_indices = frame_index[:, 1]
    valid_ca = ca_indices >= 0
    ca_mask[ca_indices[valid_ca]] = True
    backbone_mask = inputs.get("backbone_mask")
    if backbone_mask is None:
        backbone_mask = torch.ones(
            apo.shape[0], dtype=torch.bool, device=apo.device
        )
    else:
        backbone_mask = backbone_mask.to(predicted.device)
    rigid_origin, rigid_frame, rigid_valid = residue_frames(
        apo, frame_index
    )
    holo_origin, holo_frame, holo_frame_valid = residue_frames(
        holo, frame_index
    )
    bridge = remaining_transform_current_to_holo(
        rigid_origin,
        rigid_frame,
        holo_origin,
        holo_frame,
        frame_valid=rigid_valid & holo_frame_valid,
    )
    rigid_oracle = apply_motion(
        inputs,
        apo,
        bridge.translation_local,
        bridge.rotvec_local,
        torch.zeros(
            (bridge.translation_local.shape[0], NUM_CHI),
            dtype=apo.dtype,
            device=apo.device,
        ),
    )
    holo_atom_rmsd = _atom_rmsd(predicted, holo)
    apo_atom_rmsd = _atom_rmsd(apo, holo)
    holo_ca_rmsd = _masked_atom_rmsd(predicted, holo, ca_mask)
    apo_ca_rmsd = _masked_atom_rmsd(apo, holo, ca_mask)
    holo_backbone_rmsd = _masked_atom_rmsd(predicted, holo, backbone_mask)
    apo_backbone_rmsd = _masked_atom_rmsd(apo, holo, backbone_mask)
    rigid_oracle_holo_atom_rmsd = _atom_rmsd(rigid_oracle, holo)
    rigid_oracle_holo_backbone_rmsd = _masked_atom_rmsd(
        rigid_oracle, holo, backbone_mask
    )
    displacement = torch.linalg.vector_norm(predicted - apo, dim=-1)
    return {
        "sample_id": record["sample_id"],
        "finite": bool(torch.isfinite(predicted).all()),
        "pocket_atoms": int(pocket.sum().item()),
        "apo_holo_coordinate_rmse": _coord_rmse(apo, holo),
        "sample_apo_coordinate_rmse": _coord_rmse(predicted, apo),
        "sample_holo_coordinate_rmse": _coord_rmse(predicted, holo),
        "improvement_coordinate_rmse": _coord_rmse(apo, holo)
        - _coord_rmse(predicted, holo),
        "apo_holo_atom_rmsd": _atom_rmsd(apo, holo),
        "sample_apo_atom_rmsd": _atom_rmsd(predicted, apo),
        "sample_holo_atom_rmsd": holo_atom_rmsd,
        "improvement_atom_rmsd": apo_atom_rmsd - holo_atom_rmsd,
        "apo_holo_ca_rmsd": apo_ca_rmsd,
        "sample_apo_ca_rmsd": _masked_atom_rmsd(predicted, apo, ca_mask),
        "sample_holo_ca_rmsd": holo_ca_rmsd,
        "improvement_ca_rmsd": apo_ca_rmsd - holo_ca_rmsd,
        "apo_holo_backbone_rmsd": apo_backbone_rmsd,
        "sample_apo_backbone_rmsd": _masked_atom_rmsd(
            predicted, apo, backbone_mask
        ),
        "sample_holo_backbone_rmsd": holo_backbone_rmsd,
        "improvement_backbone_rmsd": apo_backbone_rmsd - holo_backbone_rmsd,
        "rigid_oracle_holo_atom_rmsd": rigid_oracle_holo_atom_rmsd,
        "rigid_oracle_holo_backbone_rmsd": rigid_oracle_holo_backbone_rmsd,
        "apo_holo_pocket_coordinate_rmse": _coord_rmse(apo[pocket], holo[pocket]),
        "sample_apo_pocket_coordinate_rmse": _coord_rmse(predicted[pocket], apo[pocket]),
        "sample_holo_pocket_coordinate_rmse": _coord_rmse(predicted[pocket], holo[pocket]),
        "pocket_improvement_coordinate_rmse": _coord_rmse(apo[pocket], holo[pocket])
        - _coord_rmse(predicted[pocket], holo[pocket]),
        "apo_holo_pocket_atom_rmsd": _atom_rmsd(apo[pocket], holo[pocket]),
        "sample_apo_pocket_atom_rmsd": _atom_rmsd(predicted[pocket], apo[pocket]),
        "sample_holo_pocket_atom_rmsd": _atom_rmsd(predicted[pocket], holo[pocket]),
        "pocket_improvement_atom_rmsd": _atom_rmsd(apo[pocket], holo[pocket])
        - _atom_rmsd(predicted[pocket], holo[pocket]),
        "direction_cosine": _direction_cosine(apo, holo, predicted),
        "sample_displacement_mean": float(displacement.mean().item()),
        "sample_displacement_max": float(displacement.max().item()),
    }


def _bootstrap_mean_ci(values, *, seed: int = 20260924, draws: int = 2000):
    values = torch.tensor(values, dtype=torch.float64)
    if values.numel() == 0:
        return [0.0, 0.0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        values.numel(), (draws, values.numel()), generator=generator
    )
    means = values[indices].mean(dim=1)
    bounds = torch.quantile(
        means,
        torch.tensor([0.025, 0.975], dtype=means.dtype),
    )
    return [float(bounds[0]), float(bounds[1])]


def _trajectory_metrics(
    record,
    trajectory,
    *,
    schedule_type: str = "remaining",
    max_fraction: float | None = None,
    fixed_fraction: float | None = None,
    damping: float = 0.25,
    disable_chi: bool = True,
):
    inputs, targets = record["input"], record["target"]
    apo = inputs["apo_pos"].to(trajectory[0].device)
    holo = targets["holo_pos"].to(trajectory[0].device)
    ligand = inputs["ligand_pos"].to(trajectory[0].device)
    pocket = torch.cdist(apo, ligand).min(dim=-1).values <= 8.0
    if not bool(pocket.any()):
        pocket = torch.ones(apo.shape[0], dtype=torch.bool, device=apo.device)
    frame_index = inputs["frame_index"].to(trajectory[0].device)
    ca_mask = torch.zeros(apo.shape[0], dtype=torch.bool, device=apo.device)
    ca_indices = frame_index[:, 1]
    valid_ca = ca_indices >= 0
    ca_mask[ca_indices[valid_ca]] = True
    backbone_mask = inputs.get("backbone_mask")
    if backbone_mask is None:
        backbone_mask = torch.ones(
            apo.shape[0], dtype=torch.bool, device=apo.device
        )
    else:
        backbone_mask = backbone_mask.to(trajectory[0].device)
    total_steps = len(trajectory) - 1
    result = []
    for step, state in enumerate(trajectory):
        row = {
            "step": step,
            "apo_atom_rmsd": _atom_rmsd(state, apo),
            "holo_atom_rmsd": _atom_rmsd(state, holo),
            "apo_ca_rmsd": _masked_atom_rmsd(state, apo, ca_mask),
            "holo_ca_rmsd": _masked_atom_rmsd(state, holo, ca_mask),
            "apo_backbone_rmsd": _masked_atom_rmsd(
                state, apo, backbone_mask
            ),
            "holo_backbone_rmsd": _masked_atom_rmsd(
                state, holo, backbone_mask
            ),
            "pocket_apo_atom_rmsd": _atom_rmsd(state[pocket], apo[pocket]),
            "pocket_holo_atom_rmsd": _atom_rmsd(state[pocket], holo[pocket]),
            "displacement_mean": float(
                torch.linalg.vector_norm(state - apo, dim=-1).mean().item()
            ),
            "direction_cosine": _direction_cosine(apo, holo, state),
        }
        if step < total_steps:
            current_origin, current_frame, current_valid = residue_frames(
                state, frame_index
            )
            holo_origin, holo_frame, holo_valid = residue_frames(
                holo, frame_index
            )
            bridge = remaining_transform_current_to_holo(
                current_origin,
                current_frame,
                holo_origin,
                holo_frame,
                frame_valid=current_valid & holo_valid,
            )
            fraction = schedule_fraction(
                total_steps - step,
                total_steps,
                schedule_type=schedule_type,
                max_fraction=max_fraction,
                fixed_fraction=fixed_fraction,
                damping=damping,
            )
            oracle_next = apply_motion(
                inputs,
                state,
                bridge.translation_local,
                bridge.rotvec_local,
                torch.zeros(
                    (bridge.translation_local.shape[0], NUM_CHI),
                    dtype=state.dtype,
                    device=state.device,
                ),
                fraction=fraction,
            )
            predicted_delta = trajectory[step + 1] - state
            oracle_delta = oracle_next - state
            denominator = oracle_delta.square().sum().clamp_min(1e-8)
            projection = (predicted_delta * oracle_delta).sum() / denominator
            orthogonal = torch.linalg.vector_norm(
                predicted_delta - projection * oracle_delta
            )
            row["oracle_delta_norm"] = float(
                torch.linalg.vector_norm(oracle_delta).item()
            )
            row["projection_scale"] = float(projection.item())
            row["orthogonal_error"] = float(orthogonal.item())
        result.append(row)
    return result


def _summarize(rows: List[Dict[str, object]]):
    finite = [row for row in rows if row["finite"]]
    if not finite:
        return {"count": len(rows), "finite_count": 0, "finite_fraction": 0.0}

    def mean(key):
        return sum(float(row[key]) for row in finite) / len(finite)

    summary = {
        "count": len(rows),
        "finite_count": len(finite),
        "finite_fraction": len(finite) / max(1, len(rows)),
        "apo_holo_coordinate_rmse_mean": mean("apo_holo_coordinate_rmse"),
        "sample_apo_coordinate_rmse_mean": mean("sample_apo_coordinate_rmse"),
        "sample_holo_coordinate_rmse_mean": mean("sample_holo_coordinate_rmse"),
        "improvement_coordinate_rmse_mean": mean("improvement_coordinate_rmse"),
        "improved_coordinate_rmse_fraction": sum(
            row["improvement_coordinate_rmse"] > 0 for row in finite
        )
        / len(finite),
        "apo_holo_atom_rmsd_mean": mean("apo_holo_atom_rmsd"),
        "sample_apo_atom_rmsd_mean": mean("sample_apo_atom_rmsd"),
        "sample_holo_atom_rmsd_mean": mean("sample_holo_atom_rmsd"),
        "improvement_atom_rmsd_mean": mean("improvement_atom_rmsd"),
        "improved_atom_rmsd_fraction": sum(row["improvement_atom_rmsd"] > 0 for row in finite)
        / len(finite),
        "apo_holo_ca_rmsd_mean": mean("apo_holo_ca_rmsd"),
        "sample_holo_ca_rmsd_mean": mean("sample_holo_ca_rmsd"),
        "improvement_ca_rmsd_mean": mean("improvement_ca_rmsd"),
        "improved_ca_rmsd_fraction": sum(
            row["improvement_ca_rmsd"] > 0 for row in finite
        )
        / len(finite),
        "apo_holo_backbone_rmsd_mean": mean("apo_holo_backbone_rmsd"),
        "sample_holo_backbone_rmsd_mean": mean("sample_holo_backbone_rmsd"),
        "improvement_backbone_rmsd_mean": mean("improvement_backbone_rmsd"),
        "improved_backbone_rmsd_fraction": sum(
            row["improvement_backbone_rmsd"] > 0 for row in finite
        )
        / len(finite),
        "rigid_oracle_holo_atom_rmsd_mean": mean(
            "rigid_oracle_holo_atom_rmsd"
        ),
        "rigid_oracle_holo_backbone_rmsd_mean": mean(
            "rigid_oracle_holo_backbone_rmsd"
        ),
        "apo_holo_pocket_coordinate_rmse_mean": mean("apo_holo_pocket_coordinate_rmse"),
        "sample_apo_pocket_coordinate_rmse_mean": mean("sample_apo_pocket_coordinate_rmse"),
        "sample_holo_pocket_coordinate_rmse_mean": mean("sample_holo_pocket_coordinate_rmse"),
        "pocket_improvement_coordinate_rmse_mean": mean("pocket_improvement_coordinate_rmse"),
        "pocket_improved_coordinate_rmse_fraction": sum(
            row["pocket_improvement_coordinate_rmse"] > 0 for row in finite
        )
        / len(finite),
        "apo_holo_pocket_atom_rmsd_mean": mean("apo_holo_pocket_atom_rmsd"),
        "sample_apo_pocket_atom_rmsd_mean": mean("sample_apo_pocket_atom_rmsd"),
        "sample_holo_pocket_atom_rmsd_mean": mean("sample_holo_pocket_atom_rmsd"),
        "pocket_improvement_atom_rmsd_mean": mean("pocket_improvement_atom_rmsd"),
        "pocket_improved_atom_rmsd_fraction": sum(
            row["pocket_improvement_atom_rmsd"] > 0 for row in finite
        )
        / len(finite),
        "direction_cosine_mean": mean("direction_cosine"),
        "sample_displacement_mean": mean("sample_displacement_mean"),
        "sample_displacement_max_mean": mean("sample_displacement_max"),
        "pocket_atoms_mean": mean("pocket_atoms"),
    }
    for metric in (
        "atom_rmsd",
        "ca_rmsd",
        "backbone_rmsd",
    ):
        improvements = [
            float(row["improvement_" + metric])
            for row in finite
        ]
        summary["improvement_" + metric + "_paired_bootstrap_95ci"] = (
            _bootstrap_mean_ci(improvements)
        )

    max_steps = max(
        (len(row.get("trajectory", [])) for row in finite),
        default=0,
    )
    trajectory_summary = []
    for step in range(max_steps):
        step_rows = [
            row["trajectory"][step]
            for row in finite
            if len(row.get("trajectory", [])) > step
        ]
        keys = (
            "apo_atom_rmsd",
            "holo_atom_rmsd",
            "apo_ca_rmsd",
            "holo_ca_rmsd",
            "apo_backbone_rmsd",
            "holo_backbone_rmsd",
            "displacement_mean",
            "direction_cosine",
            "projection_scale",
            "orthogonal_error",
            "oracle_delta_norm",
        )
        step_summary = {"step": step, "count": len(step_rows)}
        for key in keys:
            values = [
                float(row[key]) for row in step_rows if key in row
            ]
            if values:
                step_summary[key + "_mean"] = sum(values) / len(values)
        trajectory_summary.append(step_summary)
    summary["trajectory"] = trajectory_summary
    return summary


def _batches(samples, batch_size: int, max_nodes: int):
    cursor = 0
    while cursor < len(samples):
        chosen = []
        node_count = 0
        for sample in samples[cursor : cursor + batch_size]:
            sample_nodes = (
                sample["input"]["apo_pos"].shape[0]
                + sample["input"]["ligand_pos"].shape[0]
            )
            if chosen and node_count + sample_nodes > max_nodes:
                break
            chosen.append(sample)
            node_count += sample_nodes
        if not chosen:
            chosen = [samples[cursor]]
        yield chosen
        cursor += len(chosen)


def _merge_split_rows(parts, splits):
    rows_by_split = {split: [] for split in splits}
    for part in parts:
        for split in splits:
            rows_by_split[split].extend(part[split])
    return rows_by_split


def main():
    args = _arguments()
    if (
        args.batch_size <= 0
        or args.max_nodes <= 0
        or (args.steps is not None and args.steps <= 0)
        or not 0.0 <= args.motion_scale <= 1.0
        or not 0.0 <= args.initial_noise_scale <= 1.0
    ):
        raise ValueError(
            "batch-size, max-nodes, and steps must be positive; "
            "motion-scale and initial-noise-scale must be in [0, 1]"
        )

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("PocketDiff v4 evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group(backend="gloo")

    cache = load_cache(args.data)
    try:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != "pocketdiff-v4-checkpoint":
        raise ValueError("not a PocketDiff v4 training checkpoint")
    if checkpoint.get("cache_source", {}).get("sha256") != cache["source"]["sha256"]:
        raise ValueError("checkpoint/cache source fingerprints differ")
    config = checkpoint["config"]
    stored_cache = checkpoint.get("cache", {})
    if stored_cache.get("sha256"):
        current_fingerprint = source_fingerprint(args.data)
        if stored_cache["sha256"] != current_fingerprint["sha256"]:
            raise ValueError("checkpoint/cache file fingerprints differ")
    steps = args.steps or int(config.get("validation_steps", 4))
    schedule_type = args.schedule_type or config.get("schedule_type", "remaining")
    max_fraction = (
        args.max_fraction
        if args.max_fraction is not None
        else config.get("max_fraction")
    )
    fixed_fraction = (
        args.fixed_fraction
        if args.fixed_fraction is not None
        else config.get("fixed_fraction")
    )
    damping = (
        args.damping if args.damping is not None else config.get("damping", 0.25)
    )
    from .model import PocketDiffV4Model

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

    local_rows = {}
    for split in args.splits:
        samples = cache["splits"][split]
        if args.max_per_split:
            samples = samples[: args.max_per_split]
        local_samples = samples[rank::world]
        rows = []
        for chosen in _batches(local_samples, args.batch_size, args.max_nodes):
            batch = _move(collate_complexes(chosen), device)
            inputs = batch["input"]
            sample_ids = batch["target"]["sample_id"]
            predicted, trajectory = sample_complexes_with_trajectory(
                model,
                inputs,
                sample_ids,
                steps=steps,
                seed=args.seed,
                motion_scale=args.motion_scale,
                initial_noise_scale=args.initial_noise_scale,
                disable_chi=args.disable_chi,
                schedule_type=schedule_type,
                max_fraction=max_fraction,
                fixed_fraction=fixed_fraction,
                damping=damping,
            )
            atom_ptr = inputs["atom_ptr"].tolist()
            for graph_id, record in enumerate(chosen):
                start, end = atom_ptr[graph_id : graph_id + 2]
                row = _sample_metrics(record, predicted[start:end])
                row["trajectory"] = _trajectory_metrics(
                    record,
                    [state[start:end] for state in trajectory],
                    schedule_type=schedule_type,
                    max_fraction=max_fraction,
                    fixed_fraction=fixed_fraction,
                    damping=damping,
                    disable_chi=args.disable_chi,
                )
                rows.append(row)
        local_rows[split] = rows

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard = output.with_name(output.stem + ".rank%d.json" % rank)
    shard.write_text(json.dumps(local_rows, sort_keys=True) + "\n")
    if world > 1:
        dist.barrier()
    if rank == 0:
        shards = []
        for shard_rank in range(world):
            shard_path = output.with_name(
                output.stem + ".rank%d.json" % shard_rank
            )
            with shard_path.open("r") as stream:
                shards.append(json.load(stream))
        rows_by_split = _merge_split_rows(shards, args.splits)
        result = {
            "format": "pocketdiff-v4-evaluation",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_update": int(checkpoint["update"]),
            "cache_source": cache["source"],
            "split_counts": {
                split: len(cache["sample_ids"][split])
                if not args.max_per_split
                else min(args.max_per_split, len(cache["sample_ids"][split]))
                for split in args.splits
            },
            "world_size": world,
            "steps": steps,
            "seed": args.seed,
            "motion_scale": args.motion_scale,
            "initial_noise_scale": args.initial_noise_scale,
            "schedule_type": schedule_type,
            "max_fraction": max_fraction,
            "fixed_fraction": fixed_fraction,
            "damping": damping,
            "disable_chi": args.disable_chi,
            "splits": list(args.splits),
            "initialization": (
                "apo coordinates"
                if args.initial_noise_scale == 0.0
                else "apo coordinates plus sample-seeded local rigid Gaussian noise"
            ),
            "pocket_cutoff_angstrom": 8.0,
            "holo_usage": "scoring only, after apo-only sampling",
            "metrics": {
                split: _summarize(rows_by_split[split])
                for split in args.splits
            },
            "rows": {
                split: sorted(rows, key=lambda row: row["sample_id"])
                for split, rows in rows_by_split.items()
                if split in args.splits
            },
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["metrics"], indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
