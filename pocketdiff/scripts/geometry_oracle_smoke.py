"""Run the Phase 2 geometry/oracle check on a small raw Apo2Mol sample."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.geometry.bridge import (
    build_bridge_state,
    oracle_reconstruction_metrics,
    remaining_transform_current_to_holo,
)
from pocketdiff.geometry.frames import build_residue_frames, frame_orthogonality_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    with args.split_pickle.open("rb") as handle:
        records = pickle.load(handle)["train"][: args.limit]
    adapter = Apo2MolAdapter(args.data_root)
    samples = []
    failures = {}
    for record in records:
        try:
            value, adapter_diagnostics = adapter.convert_record_with_diagnostics(record)
            apo_frame = build_residue_frames(
                value.protein_pos_apo,
                value.atom_to_residue,
                value.protein_atom_name,
                num_residues=value.num_residues,
            )
            holo_frame = build_residue_frames(
                value.protein_pos_holo,
                value.atom_to_residue,
                value.protein_atom_name,
                num_residues=value.num_residues,
            )
            valid = apo_frame.valid & holo_frame.valid & value.frame_valid
            bridge_one = build_bridge_state(
                value.protein_pos_apo,
                value.atom_to_residue,
                apo_frame.origins,
                apo_frame.frames,
                holo_frame.origins,
                holo_frame.frames,
                fraction=1.0,
                frame_valid=valid,
            )
            metrics = oracle_reconstruction_metrics(
                bridge_one.protein_pos,
                value.protein_pos_holo,
                value.atom_to_residue,
                value.protein_atom_name,
                valid,
            )
            remaining = remaining_transform_current_to_holo(
                apo_frame.origins,
                apo_frame.frames,
                holo_frame.origins,
                holo_frame.frames,
                frame_valid=valid,
            )
            samples.append(
                {
                    "sample_id": value.sample_id,
                    "num_protein_atoms": value.num_protein_atoms,
                    "num_residues": value.num_residues,
                    "frame_valid": int(valid.sum()),
                    "frame_invalid": int((~valid).sum()),
                    "frame_orthogonality_error": float(frame_orthogonality_error(apo_frame.frames)),
                    "apo_frame_det_min": float(torch.linalg.det(apo_frame.frames).min()),
                    "holo_frame_det_min": float(torch.linalg.det(holo_frame.frames).min()),
                    "oracle": metrics.__dict__,
                    "remaining_translation_abs_max": float(remaining.translation_local.abs().max()),
                    "remaining_rotvec_abs_max": float(remaining.rotvec_local.abs().max()),
                    "adapter": adapter_diagnostics.to_dict(),
                }
            )
        except Exception as exc:
            failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1

    report = {"requested": len(records), "successful": len(samples), "failures": failures, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"requested": len(records), "successful": len(samples), "failures": failures}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
