from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

from .batching import collate_complexes
from .cache import load_cache
from .sampler import sample_complexes


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="pocketdiff_v4/data/residue_graphs.pt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=12000)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--initial-noise-scale", type=float, default=1.0)
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


def _rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt((left - right).square().mean()).item())


def _sample_metrics(record: Dict[str, object], predicted: torch.Tensor):
    inputs, targets = record["input"], record["target"]
    apo = inputs["apo_pos"].to(predicted.device)
    holo = targets["holo_pos"].to(predicted.device)
    ligand = inputs["ligand_pos"].to(predicted.device)
    pocket = torch.cdist(apo, ligand).min(dim=-1).values <= 8.0
    if not bool(pocket.any()):
        pocket = torch.ones(apo.shape[0], dtype=torch.bool, device=apo.device)
    displacement = torch.linalg.vector_norm(predicted - apo, dim=-1)
    return {
        "sample_id": record["sample_id"],
        "finite": bool(torch.isfinite(predicted).all()),
        "pocket_atoms": int(pocket.sum().item()),
        "apo_holo_rmsd": _rmsd(apo, holo),
        "sample_apo_rmsd": _rmsd(predicted, apo),
        "sample_holo_rmsd": _rmsd(predicted, holo),
        "improvement": _rmsd(apo, holo) - _rmsd(predicted, holo),
        "apo_holo_pocket_rmsd": _rmsd(apo[pocket], holo[pocket]),
        "sample_apo_pocket_rmsd": _rmsd(predicted[pocket], apo[pocket]),
        "sample_holo_pocket_rmsd": _rmsd(predicted[pocket], holo[pocket]),
        "pocket_improvement": _rmsd(apo[pocket], holo[pocket])
        - _rmsd(predicted[pocket], holo[pocket]),
        "sample_displacement_mean": float(displacement.mean().item()),
        "sample_displacement_max": float(displacement.max().item()),
    }


def _summarize(rows: List[Dict[str, object]]):
    finite = [row for row in rows if row["finite"]]
    if not finite:
        return {"count": len(rows), "finite_count": 0, "finite_fraction": 0.0}

    def mean(key):
        return sum(float(row[key]) for row in finite) / len(finite)

    return {
        "count": len(rows),
        "finite_count": len(finite),
        "finite_fraction": len(finite) / max(1, len(rows)),
        "apo_holo_rmsd_mean": mean("apo_holo_rmsd"),
        "sample_apo_rmsd_mean": mean("sample_apo_rmsd"),
        "sample_holo_rmsd_mean": mean("sample_holo_rmsd"),
        "improvement_mean": mean("improvement"),
        "improved_fraction": sum(row["improvement"] > 0 for row in finite)
        / len(finite),
        "apo_holo_pocket_rmsd_mean": mean("apo_holo_pocket_rmsd"),
        "sample_apo_pocket_rmsd_mean": mean("sample_apo_pocket_rmsd"),
        "sample_holo_pocket_rmsd_mean": mean("sample_holo_pocket_rmsd"),
        "pocket_improvement_mean": mean("pocket_improvement"),
        "pocket_improved_fraction": sum(
            row["pocket_improvement"] > 0 for row in finite
        )
        / len(finite),
        "sample_displacement_mean": mean("sample_displacement_mean"),
        "sample_displacement_max_mean": mean("sample_displacement_max"),
        "pocket_atoms_mean": mean("pocket_atoms"),
    }


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
        or args.steps <= 0
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
    from .model import PocketDiffV4Model

    model = PocketDiffV4Model(
        hidden=int(config["hidden"]),
        vector_channels=int(config["vector_channels"]),
        layers=int(config["layers"]),
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
            predicted = sample_complexes(
                model,
                inputs,
                sample_ids,
                steps=args.steps,
                seed=args.seed,
                motion_scale=args.motion_scale,
                initial_noise_scale=args.initial_noise_scale,
            )
            atom_ptr = inputs["atom_ptr"].tolist()
            for graph_id, record in enumerate(chosen):
                start, end = atom_ptr[graph_id : graph_id + 2]
                rows.append(_sample_metrics(record, predicted[start:end]))
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
            "steps": args.steps,
            "seed": args.seed,
            "motion_scale": args.motion_scale,
            "initial_noise_scale": args.initial_noise_scale,
            "splits": list(args.splits),
            "initialization": "apo coordinates plus sample-seeded local rigid Gaussian noise",
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
