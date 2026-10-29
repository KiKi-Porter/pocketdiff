from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.md_dataset import PreparedMDDataset
from scripts.train_md_dynamic import (
    _load_model,
    _make_loader,
    _subset_dataset,
    _validate,
)
from utils import misc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/md_dynamic.yml"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="May be supplied multiple times for baseline/checkpoint comparison.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = misc.load_config(str(args.config))
    cache_path = args.cache or Path(config.data.cache)
    dataset = PreparedMDDataset(cache_path, split=args.split)
    dataset = _subset_dataset(dataset, args.max_samples)
    loader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device)
    time_steps = [int(value) for value in config.train.val_time_steps]

    reports = []
    for checkpoint_path in args.checkpoint:
        model, payload = _load_model(config, checkpoint_path, device)
        metrics = _validate(
            model,
            loader,
            device,
            time_steps,
            args.max_batches,
        )
        reports.append(
            {
                "checkpoint": str(checkpoint_path),
                "source_iteration": payload.get("iteration"),
                "stage": payload.get("stage", "original"),
                "sample_count": len(dataset),
                "split": args.split,
                **metrics,
            }
        )

    result = {
        "config": str(args.config),
        "cache": str(cache_path),
        "time_steps": time_steps,
        "reports": reports,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
