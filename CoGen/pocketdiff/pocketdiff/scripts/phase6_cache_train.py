"""Small clean train/holdout smoke using only verified `.pt` samples."""

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
from pocketdiff.preprocessing import (
    load_cached_clean_examples,
    load_manifest,
    save_sample_cache,
    write_manifest,
)
from pocketdiff.training import collate_clean_examples, masked_remaining_motion_loss, save_checkpoint, train_clean_batch


def _rmsd(predicted, target, batch) -> float:
    valid_atom = batch.frame_valid[batch.atom_to_residue_global]
    error_sq = (predicted - target).square().sum(dim=-1)
    return float(torch.sqrt(error_sq[valid_atom].mean()).item())


def _build_cache(records, adapter, root: Path, output_parent: Path, limit: int):
    cache_dir = output_parent / "cache_samples"
    entries = []
    filtered = {}
    for record in records:
        try:
            value = adapter.convert_record(record)
            cache_path = cache_dir / f"{value.sample_id.replace('/', '_')}.pt"
            entry = save_sample_cache(
                cache_path,
                value,
                source_paths={
                    "holo_pocket": root / record[0],
                    "apo_pocket": root / record[1],
                    "ligand": root / record[2],
                },
                split="train",
            )
            entry["cache_path"] = str(cache_path.relative_to(output_parent))
            entries.append(entry)
        except AdapterError as exc:
            filtered[exc.reason_code] = filtered.get(exc.reason_code, 0) + 1
        if len(entries) >= limit:
            break
    return entries, filtered


def _predict_positions(model, batch):
    model.eval()
    with torch.no_grad():
        prediction = model(**batch.model_kwargs())
        current_frames = build_residue_frames(
            batch.protein_pos,
            batch.atom_to_residue_global,
            batch.protein_atom_name,
            num_residues=batch.residue_type.shape[0],
        )
        predicted = apply_fractional_update(
            batch.protein_pos,
            batch.atom_to_residue_global,
            current_frames.origins,
            current_frames.frames,
            prediction.remaining_translation_local,
            prediction.remaining_rotvec_local,
            remaining_steps=1,
            frame_valid=prediction.frame_valid,
        )
    return prediction, predicted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_samples <= args.holdout or args.holdout <= 0 or args.steps <= 0:
        parser.error("require num_samples > holdout > 0 and steps > 0")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.split_pickle.open("rb") as handle:
        raw_records = pickle.load(handle)["train"]
    adapter = Apo2MolAdapter(args.data_root)
    entries, filtered = _build_cache(raw_records, adapter, args.data_root, args.output.parent, args.num_samples)
    if len(entries) <= args.holdout:
        raise RuntimeError("not enough valid cached samples for train/holdout")
    manifest_path = args.output.parent / "manifest.json"
    write_manifest(
        manifest_path,
        entries,
        config={"split": "train", "num_samples": args.num_samples, "holdout": args.holdout, "cache_edge_policy": "dynamic_edges_not_cached"},
        filtered_counts=filtered,
    )
    manifest = load_manifest(manifest_path)
    examples = load_cached_clean_examples(manifest_path, verify_sources=True)
    train_examples = examples[:-args.holdout]
    holdout_examples = examples[-args.holdout:]
    train_batch = collate_clean_examples(train_examples)
    holdout_batch = collate_clean_examples(holdout_examples)
    model_config = {"encoder_layers": 1, "knn": 8, "sigma_translation": 1.0}
    model = PocketDiffModel(**model_config)
    model.eval()
    with torch.no_grad():
        initial_train_prediction = model(**train_batch.model_kwargs())
        initial_holdout_prediction = model(**holdout_batch.model_kwargs())
        initial_train_loss = masked_remaining_motion_loss(initial_train_prediction, train_batch.target_translation_local, train_batch.target_rotvec_local, train_batch.frame_valid).loss.item()
        initial_holdout_loss = masked_remaining_motion_loss(initial_holdout_prediction, holdout_batch.target_translation_local, holdout_batch.target_rotvec_local, holdout_batch.frame_valid).loss.item()
        initial_holdout_rmsd = _rmsd(holdout_batch.protein_pos, holdout_batch.protein_pos_holo, holdout_batch)
    history = train_clean_batch(model, train_batch, steps=args.steps, learning_rate=args.learning_rate, log_every=max(1, args.steps // 10))
    final_train_prediction, final_train_positions = _predict_positions(model, train_batch)
    final_holdout_prediction, final_holdout_positions = _predict_positions(model, holdout_batch)
    final_train_loss = masked_remaining_motion_loss(final_train_prediction, train_batch.target_translation_local, train_batch.target_rotvec_local, train_batch.frame_valid).loss.item()
    final_holdout_loss = masked_remaining_motion_loss(final_holdout_prediction, holdout_batch.target_translation_local, holdout_batch.target_rotvec_local, holdout_batch.frame_valid).loss.item()
    final_holdout_rmsd = _rmsd(final_holdout_positions, holdout_batch.protein_pos_holo, holdout_batch)
    checkpoint_path = args.output.with_suffix(".pt")
    save_checkpoint(model= model, path=checkpoint_path, config=model_config, sample_ids=train_batch.sample_ids, final_record=history[-1])
    restored = PocketDiffModel(**model_config)
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        restored_prediction = restored(**train_batch.model_kwargs())
    reload_max_translation_error = float(
        (restored_prediction.remaining_translation_local - final_train_prediction.remaining_translation_local).abs().max()
    )
    report = {
        "manifest": str(manifest_path),
        "manifest_format": manifest["format"],
        "requested_samples": args.num_samples,
        "cached_samples": len(examples),
        "train_samples": len(train_examples),
        "holdout_samples": len(holdout_examples),
        "train_ids": train_batch.sample_ids,
        "holdout_ids": holdout_batch.sample_ids,
        "filtered_counts": filtered,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_holdout_loss": initial_holdout_loss,
        "final_holdout_loss": final_holdout_loss,
        "initial_holdout_rmsd": initial_holdout_rmsd,
        "final_holdout_rmsd": final_holdout_rmsd,
        "checkpoint": str(checkpoint_path),
        "reload_max_translation_error": reload_max_translation_error,
        "history": [record.__dict__ for record in history],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("cached_samples", "train_samples", "holdout_samples", "initial_train_loss", "final_train_loss", "initial_holdout_loss", "final_holdout_loss", "initial_holdout_rmsd", "final_holdout_rmsd")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
