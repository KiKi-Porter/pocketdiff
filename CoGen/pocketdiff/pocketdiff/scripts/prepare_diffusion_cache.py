"""Build the reusable v4.6 CPU training cache.

The input may be an existing PocketDiff cache or the raw Apo2Mol split. Raw
records are converted exactly once here; training ranks only deserialize the
resulting CPU cache and never repeat PDB/SDF parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.preprocessing.diffusion_cache import save_diffusion_cache
from pocketdiff.training.diffusion import (
    load_split_records,
    select_convertible_complexes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-cache", type=Path)
    source.add_argument(
        "--data-root",
        type=Path,
        help="Raw Apo2Mol data root; used when --source-cache is omitted.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--valid-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--train-seed", type=int, default=5700)
    parser.add_argument("--holdout-seed", type=int, default=5701)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Write raw-preparation progress every N records.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-steps", type=int, default=20)
    args = parser.parse_args()

    import torch

    # Raw conversion is dominated by many small PDB/RDKit/geometry operations.
    # A large BLAS thread pool makes this phase slower and can exhaust process
    # resources when several independent operations are in flight.
    torch.set_num_threads(int(os.environ.get("POCKETDIFF_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(
        int(os.environ.get("POCKETDIFF_TORCH_INTEROP_THREADS", "1"))
    )

    rejected = {}
    if args.source_cache is not None:
        payload = torch.load(args.source_cache, map_location="cpu", weights_only=False)
        source_format = payload.get("format")
        if source_format == "pocketdiff-medium-cache-v1":
            values = {
                "train": payload["values"].get("train", []),
                "valid": payload["values"].get("valid", []),
                "test": payload["values"].get("test", []),
            }
        elif source_format in {
            "pocketdiff-diffusion-v45-cache-v1",
            "pocketdiff-diffusion-v46-cache-v1",
        }:
            values = payload["splits"]
        else:
            raise ValueError(
                "source cache must be pocketdiff-medium-cache-v1 or a diffusion cache"
            )
    else:
        if args.data_root is None:
            raise ValueError(
                "provide --source-cache or --data-root for raw Apo2Mol preparation"
            )
        if args.progress_every <= 0:
            raise ValueError("--progress-every must be positive")
        records = {
            split: load_split_records(args.split_path, split)
            for split in (args.train_split, args.valid_split, args.test_split)
        }
        counts = {
            "train": len(records[args.train_split])
            if args.train_count is None
            else args.train_count,
            "valid": len(records[args.valid_split])
            if args.valid_count is None
            else args.valid_count,
            "test": len(records[args.test_split])
            if args.test_count is None
            else args.test_count,
        }
        if any(count < 0 for count in counts.values()):
            raise ValueError("raw preparation counts must be non-negative")
        adapter = Apo2MolAdapter(args.data_root)
        values = {}
        for output_split, input_split, seed in (
            ("train", args.train_split, args.train_seed),
            ("valid", args.valid_split, args.holdout_seed),
            ("test", args.test_split, args.holdout_seed + 1),
        ):
            requested = args.train_count if output_split == "train" else (
                args.valid_count if output_split == "valid" else args.test_count
            )
            if requested is not None:
                result = select_convertible_complexes(
                    adapter,
                    records[input_split],
                    seed=seed,
                    count=counts[output_split],
                )
                values[output_split] = result.values
                rejected[output_split] = result.rejected
                continue

            order = list(range(len(records[input_split])))
            random.Random(seed).shuffle(order)
            converted = []
            rejected_rows = []
            for seen, index in enumerate(order, start=1):
                try:
                    converted.append(adapter.convert_record(records[input_split][index]))
                except Exception as exc:
                    rejected_rows.append(
                        {
                            "index": index,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                if seen % args.progress_every == 0 or seen == len(order):
                    progress = {
                        "format": "pocketdiff-raw-cache-progress-v1",
                        "split": output_split,
                        "processed": seen,
                        "total": len(order),
                        "converted": len(converted),
                        "rejected": len(rejected_rows),
                    }
                    args.output.with_suffix(".progress.json").write_text(
                        json.dumps(progress, indent=2, sort_keys=True) + "\n"
                    )
                    print(json.dumps(progress), flush=True)
            values[output_split] = converted
            rejected[output_split] = rejected_rows

    manifest = save_diffusion_cache(
        args.output,
        values,
        source_cache=(
            str(args.source_cache)
            if args.source_cache is not None
            else f"raw:{args.split_path.resolve()}"
        ),
        fixed_steps=args.fixed_steps,
    )
    manifest["rejected_counts"] = {
        split: len(items) for split, items in rejected.items()
    }
    manifest["rejected"] = rejected
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    args.output.with_suffix(".progress.json").write_text(
        json.dumps(
            {
                "format": "pocketdiff-raw-cache-progress-v1",
                "status": "complete",
                "counts": manifest["counts"],
                "rejected_counts": manifest["rejected_counts"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
