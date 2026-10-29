"""Reproducible clean train/holdout benchmark using verified `.pt` caches."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from pocketdiff.training import (
    CleanBatch,
    CleanSplit,
    collate_clean_examples,
    deterministic_clean_split,
    masked_remaining_motion_loss,
    save_checkpoint,
    train_clean_batch,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _rmsd(predicted: torch.Tensor, target: torch.Tensor, batch: CleanBatch) -> float:
    valid_atom = batch.frame_valid[batch.atom_to_residue_global]
    error_sq = (predicted - target).square().sum(dim=-1)
    if not bool(valid_atom.any()):
        raise ValueError("RMSD has no valid atoms")
    return float(torch.sqrt(error_sq[valid_atom].mean()).item())


def _predict_positions(model: PocketDiffModel, batch: CleanBatch):
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


def _build_cache(
    records: Iterable[Tuple[str, str, str]],
    adapter: Apo2MolAdapter,
    root: Path,
    output_parent: Path,
    limit: int,
):
    cache_dir = output_parent / "cache_samples"
    entries: List[Dict[str, object]] = []
    filtered: Dict[str, int] = {}
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


def _batch_metrics(model: PocketDiffModel, batch: CleanBatch) -> Dict[str, float]:
    prediction, predicted = _predict_positions(model, batch)
    loss = masked_remaining_motion_loss(
        prediction,
        batch.target_translation_local,
        batch.target_rotvec_local,
        batch.frame_valid,
    )
    return {
        "loss": float(loss.loss.item()),
        "translation_loss": float(loss.translation_loss.item()),
        "rotation_loss": float(loss.rotation_loss.item()),
        "atom_rmsd": _rmsd(predicted, batch.protein_pos_holo, batch),
    }


def _split_report(split: CleanSplit) -> Dict[str, object]:
    train_ids = split.train_ids
    holdout_ids = split.holdout_ids
    overlap = sorted(set(train_ids).intersection(holdout_ids))
    if overlap:
        raise RuntimeError(f"train/holdout overlap: {overlap}")
    return {
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "permutation": split.permutation,
        "overlap": overlap,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"))
    parser.add_argument("--split-pickle", type=Path, default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--holdout", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_samples <= args.holdout or args.holdout <= 0 or args.steps <= 0:
        parser.error("require num_samples > holdout > 0 and steps > 0")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("learning-rate must be positive and weight-decay non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    with args.split_pickle.open("rb") as handle:
        raw_records = pickle.load(handle)["train"]
    adapter = Apo2MolAdapter(args.data_root)
    entries, filtered = _build_cache(raw_records, adapter, args.data_root, args.output.parent, args.num_samples)
    if len(entries) != args.num_samples:
        raise RuntimeError(f"requested {args.num_samples} valid samples, got {len(entries)}")
    manifest_path = args.output.parent / "manifest.json"
    write_manifest(
        manifest_path,
        entries,
        config={
            "split": "train",
            "num_samples": args.num_samples,
            "holdout": args.holdout,
            "seed": args.seed,
            "cache_edge_policy": "dynamic_edges_not_cached",
        },
        filtered_counts=filtered,
    )
    manifest = load_manifest(manifest_path)
    examples = load_cached_clean_examples(manifest_path, verify_sources=True)
    split = deterministic_clean_split(examples, holdout=args.holdout, seed=args.seed)
    split_report = _split_report(split)
    train_batch = collate_clean_examples(split.train)
    holdout_batch = collate_clean_examples(split.holdout)
    model_config = {"encoder_layers": 1, "knn": 8, "sigma_translation": 1.0}
    model = PocketDiffModel(**model_config)
    initial_train = _batch_metrics(model, train_batch)
    initial_holdout = _batch_metrics(model, holdout_batch)
    history = train_clean_batch(
        model,
        train_batch,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_every=max(1, args.steps // 10),
    )
    final_train = _batch_metrics(model, train_batch)
    final_holdout = _batch_metrics(model, holdout_batch)
    checkpoint_path = args.output.with_suffix(".pt")
    save_checkpoint(
        model=model,
        path=checkpoint_path,
        config=model_config,
        sample_ids=train_batch.sample_ids,
        final_record=history[-1],
    )
    restored = PocketDiffModel(**model_config)
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        restored_prediction = restored(**train_batch.model_kwargs())
        final_prediction = model(**train_batch.model_kwargs())
    reload_max_translation_error = float(
        (restored_prediction.remaining_translation_local - final_prediction.remaining_translation_local).abs().max()
    )
    report = {
        "manifest": str(manifest_path),
        "manifest_format": manifest["format"],
        "requested_samples": args.num_samples,
        "cached_samples": len(examples),
        "train_samples": len(split.train),
        "holdout_samples": len(split.holdout),
        "filtered_counts": filtered,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "steps": args.steps,
        "split": split_report,
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_holdout": initial_holdout,
        "final_holdout": final_holdout,
        "checkpoint": str(checkpoint_path),
        "reload_max_translation_error": reload_max_translation_error,
        "history": [record.__dict__ for record in history],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "cached_samples": report["cached_samples"],
        "train_samples": report["train_samples"],
        "holdout_samples": report["holdout_samples"],
        "initial_holdout": initial_holdout,
        "final_holdout": final_holdout,
        "reload_max_translation_error": reload_max_translation_error,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
