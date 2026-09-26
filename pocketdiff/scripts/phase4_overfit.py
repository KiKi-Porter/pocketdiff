"""Clean teacher-forced overfit smoke for the independent PocketDiff MVP."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import AdapterError, Apo2MolAdapter
from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import (
    collate_clean_examples,
    make_clean_example,
    masked_remaining_motion_loss,
    save_checkpoint,
    train_clean_batch,
)


def _masked_atom_rmsd(predicted: torch.Tensor, target: torch.Tensor, batch) -> float:
    valid_atom = batch.frame_valid[batch.atom_to_residue_global]
    error_sq = (predicted - target).square().sum(dim=-1)
    return float(torch.sqrt(error_sq[valid_atom].mean()).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.steps <= 0:
        parser.error("--num-samples and --steps must be positive")

    with args.split_pickle.open("rb") as handle:
        records = pickle.load(handle)["train"]
    adapter = Apo2MolAdapter(args.data_root)
    examples = []
    failures = {}
    for record in records:
        try:
            examples.append(make_clean_example(adapter.convert_record(record)))
        except AdapterError as exc:
            failures[exc.reason_code] = failures.get(exc.reason_code, 0) + 1
        if len(examples) >= args.num_samples:
            break
    if not examples:
        raise RuntimeError("no valid clean examples were found")

    batch = collate_clean_examples(examples)
    model_config = {"encoder_layers": 1, "knn": 8, "sigma_translation": 1.0}
    model = PocketDiffModel(**model_config)
    model.eval()
    with torch.no_grad():
        initial_prediction = model(**batch.model_kwargs())
        initial_loss = masked_remaining_motion_loss(
            initial_prediction,
            batch.target_translation_local,
            batch.target_rotvec_local,
            batch.frame_valid,
        ).loss.item()
        initial_rmsd = _masked_atom_rmsd(batch.protein_pos, batch.protein_pos_holo, batch)

    records_history = train_clean_batch(
        model,
        batch,
        steps=args.steps,
        learning_rate=args.learning_rate,
        log_every=max(1, args.steps // 10),
    )
    model.eval()
    with torch.no_grad():
        final_prediction = model(**batch.model_kwargs())
        final_loss = masked_remaining_motion_loss(
            final_prediction,
            batch.target_translation_local,
            batch.target_rotvec_local,
            batch.frame_valid,
        ).loss.item()
        current_frames = build_residue_frames(
            batch.protein_pos,
            batch.atom_to_residue_global,
            batch.protein_atom_name,
            num_residues=batch.residue_type.shape[0],
        )
        predicted_pos = apply_fractional_update(
            batch.protein_pos,
            batch.atom_to_residue_global,
            current_frames.origins,
            current_frames.frames,
            final_prediction.remaining_translation_local,
            final_prediction.remaining_rotvec_local,
            remaining_steps=1,
            frame_valid=final_prediction.frame_valid,
        )
        final_rmsd = _masked_atom_rmsd(predicted_pos, batch.protein_pos_holo, batch)

    checkpoint_path = args.output.with_suffix(".pt")
    save_checkpoint(
        checkpoint_path,
        model,
        config=model_config,
        sample_ids=batch.sample_ids,
        final_record=records_history[-1],
    )
    restored = PocketDiffModel(**model_config)
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        restored_prediction = restored(**batch.model_kwargs())
    reload_max_error = float(
        (restored_prediction.remaining_translation_local - final_prediction.remaining_translation_local).abs().max()
    )
    report = {
        "model_config": model_config,
        "requested_samples": args.num_samples,
        "used_samples": len(examples),
        "sample_ids": batch.sample_ids,
        "adapter_filter_counts_before_target": failures,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / max(initial_loss, 1.0e-12),
        "identity_atom_rmsd": initial_rmsd,
        "predicted_atom_rmsd": final_rmsd,
        "rmsd_improved": final_rmsd <= initial_rmsd,
        "checkpoint": str(checkpoint_path),
        "reload_max_translation_error": reload_max_error,
        "history": [record.__dict__ for record in records_history],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("used_samples", "initial_loss", "final_loss", "identity_atom_rmsd", "predicted_atom_rmsd", "rmsd_improved")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
