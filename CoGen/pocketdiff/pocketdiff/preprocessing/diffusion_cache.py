"""Single-file v4.5 diffusion cache.

The cache deliberately stores canonical CPU ``PocketComplex`` objects rather
than serialized CUDA graphs.  Graph construction is deterministic and cheap
for the current scalar encoder, while the expensive adapter/PDB work happens
once before distributed training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from pocketdiff.data.schema import PocketComplex
from pocketdiff.training.diffusion import _build_motion_buckets


CACHE_FORMAT = "pocketdiff-diffusion-v46-cache-v1"
CACHE_VERSION = "v46-bridge-velocity-static-topology-v1"
PREPROCESSING_CONTRACT = (
    "protein27-ligand13-kabsch-frame-index-sequence-metadata-no-dynamic-graph"
)


def _frame_indices(value: PocketComplex) -> torch.Tensor:
    if value.frame_atom_indices is not None:
        return value.frame_atom_indices.detach().cpu().clone()
    result = torch.full((value.num_residues, 3), -1, dtype=torch.long)
    slots = {"N": 0, "CA": 1, "C": 2}
    for atom_index, (name, residue_id) in enumerate(
        zip(value.protein_atom_name, value.atom_to_residue.detach().cpu().tolist())
    ):
        slot = slots.get(name)
        if slot is not None and result[residue_id, slot] < 0:
            result[residue_id, slot] = atom_index
    return result


def _with_static_topology(value: PocketComplex) -> PocketComplex:
    indices = _frame_indices(value)
    if value.frame_atom_indices is not None and torch.equal(
        value.frame_atom_indices.detach().cpu(), indices
    ):
        return value
    return replace(value, frame_atom_indices=indices)


def _sample_signature(value: PocketComplex) -> Dict[str, object]:
    return {
        "sample_id": value.sample_id,
        "num_protein_atoms": value.num_protein_atoms,
        "num_residues": value.num_residues,
        "num_ligand_atoms": value.num_ligand_atoms,
        "frame_atom_indices": _frame_indices(value).tolist(),
    }


def _cache_hash(values: Mapping[str, Sequence[PocketComplex]]) -> str:
    digest = hashlib.sha256()
    for split in ("train", "valid", "test"):
        for value in values.get(split, ()):
            digest.update(json.dumps(_sample_signature(value), sort_keys=True).encode())
            digest.update(value.protein_pos_apo.numpy().tobytes())
            digest.update(value.protein_pos_holo.numpy().tobytes())
            digest.update(value.protein_feature.numpy().tobytes())
            digest.update(value.ligand_pos_ref.numpy().tobytes())
    return digest.hexdigest()


def save_diffusion_cache(
    path: Path,
    values: Mapping[str, Sequence[PocketComplex]],
    *,
    source_cache: str,
    fixed_steps: int = 20,
) -> Dict[str, object]:
    """Write a CPU-only cache and return its manifest."""

    if fixed_steps <= 0:
        raise ValueError("fixed_steps must be positive")
    normalized = {
        split: [_with_static_topology(value) for value in values.get(split, ())]
        for split in ("train", "valid", "test")
    }
    buckets, scores = _build_motion_buckets(
        normalized["train"], 4
    )
    payload = {
        "format": CACHE_FORMAT,
        "cache_version": CACHE_VERSION,
        "source_cache": str(source_cache),
        "fixed_steps": int(fixed_steps),
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "dynamic_geometry_cached": False,
        "splits": normalized,
        "motion_scores": [float(value) for value in scores],
        "motion_bucket_indices": buckets,
        "sample_signatures": {
            split: [_sample_signature(value) for value in values_list]
            for split, values_list in normalized.items()
        },
    }
    payload["cache_hash"] = _cache_hash(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "format": CACHE_FORMAT,
        "cache_version": CACHE_VERSION,
        "cache_hash": payload["cache_hash"],
        "counts": {split: len(items) for split, items in normalized.items()},
        "fixed_steps": int(fixed_steps),
        "path": str(path),
    }


def load_diffusion_cache(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        raise ValueError(f"unsupported v4.6 diffusion cache: {path}")
    if payload.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"cache version mismatch: {payload.get('cache_version')!r} "
            f"!= {CACHE_VERSION!r}"
        )
    if payload.get("preprocessing_contract") != PREPROCESSING_CONTRACT:
        raise ValueError("diffusion cache preprocessing contract mismatch")
    for split in ("train", "valid", "test"):
        if split not in payload.get("splits", {}):
            raise ValueError(f"diffusion cache is missing split {split!r}")
    expected = _cache_hash(payload["splits"])
    if expected != payload.get("cache_hash"):
        raise ValueError("v4.5 diffusion cache hash mismatch")
    return payload


__all__ = [
    "CACHE_FORMAT",
    "CACHE_VERSION",
    "load_diffusion_cache",
    "save_diffusion_cache",
]
