"""Build and verify a small, versioned raw Apo2Mol cache."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from pocketdiff.data.apo2mol_adapter import AdapterError, Apo2MolAdapter
from pocketdiff.preprocessing import load_manifest, load_sample_cache, save_sample_cache, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    with args.split_pickle.open("rb") as handle:
        records = pickle.load(handle)["train"]
    adapter = Apo2MolAdapter(args.data_root)
    cache_dir = args.output.parent / "cache_samples"
    entries = []
    filtered = {}
    for record in records:
        try:
            value = adapter.convert_record(record)
            sample_id = value.sample_id.replace("/", "_")
            cache_path = cache_dir / f"{sample_id}.pt"
            entry = save_sample_cache(
                    cache_path,
                    value,
                    source_paths={
                        "holo_pocket": args.data_root / record[0],
                        "apo_pocket": args.data_root / record[1],
                        "ligand": args.data_root / record[2],
                    },
                    split="train",
                )
            entry["cache_path"] = str(cache_path.relative_to(args.output.parent))
            entries.append(entry)
        except AdapterError as exc:
            filtered[exc.reason_code] = filtered.get(exc.reason_code, 0) + 1
        if len(entries) >= args.limit:
            break

    manifest_path = args.output.parent / "manifest.json"
    write_manifest(
        manifest_path,
        entries,
        config={"split": "train", "limit": args.limit, "cache_edge_policy": "dynamic_edges_not_cached"},
        filtered_counts=filtered,
    )
    loaded = []
    for entry in entries:
        cached = load_sample_cache(manifest_path.parent / entry["cache_path"], verify_sources=True)
        loaded.append(cached.complex_value.sample_id)
    manifest = load_manifest(manifest_path)
    report = {
        "requested": args.limit,
        "written": len(entries),
        "loaded_and_verified": len(loaded),
        "sample_ids": loaded,
        "filtered_counts": filtered,
        "manifest": str(manifest_path),
        "cache_dir": str(cache_dir),
        "manifest_format": manifest["format"],
        "schema_version": manifest["schema_version"],
        "geometry_version": manifest["geometry_version"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("requested", "written", "loaded_and_verified", "filtered_counts")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
