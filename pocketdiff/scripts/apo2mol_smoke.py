"""Small, reproducible Phase 1 raw-adapter smoke over Apo2Mol splits."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from pocketdiff.data.apo2mol_adapter import AdapterError, Apo2MolAdapter


def _run_split(adapter: Apo2MolAdapter, records: Iterable[object], limit: int) -> Dict[str, object]:
    counts: Counter = Counter()
    diagnostics: List[dict] = []
    selected_records = list(records)[:limit]
    for record in selected_records:
        try:
            _, diagnostic = adapter.convert_record_with_diagnostics(record)
            counts["ok"] += 1
            diagnostics.append(diagnostic.to_dict())
        except AdapterError as exc:
            counts[exc.reason_code] += 1
        except Exception as exc:  # keep unexpected failures visible in the report
            counts[f"unexpected:{type(exc).__name__}"] += 1

    result: Dict[str, object] = {
        "requested": len(selected_records),
        "counts": dict(sorted(counts.items())),
        "successful_diagnostics": diagnostics,
    }
    if diagnostics:
        result["aligned_calpha_rmsd_min"] = min(item["aligned_calpha_rmsd"] for item in diagnostics)
        result["aligned_calpha_rmsd_max"] = max(item["aligned_calpha_rmsd"] for item in diagnostics)
        result["normalized_by_paired_order"] = sum(
            bool(item["normalized_by_paired_order"]) for item in diagnostics
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"),
    )
    parser.add_argument(
        "--split-pickle",
        type=Path,
        default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    with args.split_pickle.open("rb") as handle:
        splits = pickle.load(handle)
    adapter = Apo2MolAdapter(args.data_root)
    report = {
        "adapter": "pocketdiff.data.apo2mol_adapter.Apo2MolAdapter",
        "data_root": str(args.data_root),
        "split_pickle": str(args.split_pickle),
        "limit_per_split": args.limit,
        "splits": {
            split: _run_split(adapter, records, args.limit)
            for split, records in splits.items()
            if split in ("train", "valid", "test")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({split: value["counts"] for split, value in report["splits"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
