"""Fine-tune TargetDiff Internal Cross on PocketDiff bridge protein states."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.bridge_finetune import BridgeStateDataset, TargetDiffInputDataset
from datasets.md_dataset import PreparedMDDataset
from scripts.train_md_dynamic import (
    _load_model,
    _make_loader,
    _run_stage,
    _DiffusionLossWrapper,
    _set_rng_seed,
    _subset_dataset,
)
from utils import misc


def _spaced_subset(dataset, count: int):
    if count >= len(dataset):
        return dataset
    ids = torch.linspace(0, len(dataset) - 1, count).round().long().tolist()
    return Subset(dataset, ids)


class DistributedWeightedSampler(Sampler[int]):
    """Shard one deterministic weighted sample stream across DDP ranks."""

    def __init__(self, weights, num_samples, num_replicas, rank, seed):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        if num_samples % self.num_replicas:
            raise ValueError("num_samples must be divisible by world size")
        self.total_samples = int(num_samples)
        self.local_samples = self.total_samples // self.num_replicas
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_samples,
            replacement=True,
            generator=generator,
        )
        return iter(indices[self.rank :: self.num_replicas].tolist())

    def __len__(self):
        return self.local_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full_dynamic_internal_cross.yml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/full_dynamic_internal_cross_20261003/best.pt")
    parser.add_argument("--bridge-cache", type=Path, default=ROOT / "data/bridge_finetune_3000.pt")
    parser.add_argument("--md-cache", type=Path, default=ROOT / "data/full_dynamic/prepared_targetdiff_full.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/bridge_finetune_3000_20261005")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-6)
    parser.add_argument(
        "--val-every",
        type=int,
        default=None,
        help="Run validation and checkpoint selection every N updates.",
    )
    parser.add_argument("--bridge-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--valid-bridge-states", type=int, default=300)
    parser.add_argument("--valid-md-samples", type=int, default=150)
    parser.add_argument("--max-bridge-train-states", type=int, default=None)
    parser.add_argument("--max-md-train-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.bridge_fraction < 1.0:
        parser.error("bridge-fraction must be between 0 and 1")
    if args.smoke:
        args.iterations = 2
        args.batch_size = 2
        args.learning_rate = 1.0e-6
        args.valid_bridge_states = 2
        args.valid_md_samples = 2
        args.max_bridge_train_states = 4
        args.max_md_train_samples = 4
        args.max_val_batches = 1
        args.output_dir = ROOT / "runs/bridge_finetune_smoke_20261005"
        args.md_cache = ROOT / "data/md300/prepared_targetdiff_md300_10frames.pt"

    _set_rng_seed(args.seed)
    distributed = _is_distributed()
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed fine-tuning requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(args.device)
    config = misc.load_config(str(args.config))
    config.train.batch_size = args.batch_size
    config.train.num_workers = args.num_workers
    config.train.pos_noise_std = 0.0
    config.train.val_every = (
        args.iterations if args.val_every is None else max(1, args.val_every)
    )
    config.train.seed = args.seed
    config.train.internal_adapter.max_iters = args.iterations
    config.train.internal_adapter.lr = args.learning_rate

    bridge_payload = torch.load(
        str(args.bridge_cache), map_location="cpu", weights_only=False
    )
    md_payload = torch.load(
        str(args.md_cache), map_location="cpu", weights_only=False
    )
    bridge_train = TargetDiffInputDataset(
        BridgeStateDataset.from_payload(bridge_payload, "train")
    )
    md_train = TargetDiffInputDataset(
        PreparedMDDataset.from_payload(md_payload, split="train")
    )
    bridge_valid = TargetDiffInputDataset(
        BridgeStateDataset.from_payload(bridge_payload, "valid")
    )
    md_valid = TargetDiffInputDataset(
        PreparedMDDataset.from_payload(md_payload, split="valid")
    )
    bridge_train = _subset_dataset(bridge_train, args.max_bridge_train_states)
    md_train = _subset_dataset(md_train, args.max_md_train_samples)
    if not len(bridge_train) or not len(md_train):
        raise ValueError("Both bridge and real MD training sources must be non-empty")

    train_dataset = ConcatDataset((bridge_train, md_train))
    weights = torch.cat((
        torch.full((len(bridge_train),), args.bridge_fraction / len(bridge_train), dtype=torch.double),
        torch.full((len(md_train),), (1.0 - args.bridge_fraction) / len(md_train), dtype=torch.double),
    ))
    if distributed:
        sampler = DistributedWeightedSampler(
            weights,
            num_samples=args.iterations * args.batch_size * world_size,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=args.iterations * args.batch_size,
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    bridge_valid = _spaced_subset(bridge_valid, args.valid_bridge_states)
    md_valid = _spaced_subset(md_valid, args.valid_md_samples)
    valid_loader = _make_loader(
        ConcatDataset((bridge_valid, md_valid)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=(
            DistributedSampler(
                ConcatDataset((bridge_valid, md_valid)),
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            if distributed
            else None
        ),
    )
    if len(valid_loader) == 0:
        raise ValueError("Validation mixture is empty")

    model, source_payload = _load_model(config, args.checkpoint, device)
    if distributed:
        model = DistributedDataParallel(
            _DiffusionLossWrapper(model),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    manifest = {
        "protocol": "targetdiff_bridge_finetune_v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "bridge_cache": str(args.bridge_cache.resolve()),
        "md_replay_cache": str(args.md_cache.resolve()),
        "source_checkpoint_iteration": source_payload.get("iteration"),
        "trainable_stage": "internal_adapter",
        "frozen_pocketdiff": True,
        "bridge_fraction": args.bridge_fraction,
        "md_fraction": 1.0 - args.bridge_fraction,
        "crossdock_replay": "unavailable in clean release; original-function anchor regularization retained",
        "train_bridge_states": len(bridge_train),
        "train_md_samples": len(md_train),
        "valid_bridge_states": len(bridge_valid),
        "valid_md_samples": len(md_valid),
        "iterations": args.iterations,
        "val_every": config.train.val_every,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "distributed": distributed,
        "world_size": world_size,
        "local_batch_size": args.batch_size,
    }
    if rank == 0:
        (args.output_dir / "bridge_finetune_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    stage_summary = _run_stage(
        model=model,
        stage="internal_adapter",
        stage_config=config.train.internal_adapter,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=args.output_dir,
        config=config,
        source_checkpoint=args.checkpoint,
        max_val_batches=args.max_val_batches,
    )
    summary = {"manifest": manifest, "stage": stage_summary}
    if rank == 0:
        (args.output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
