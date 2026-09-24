from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch

from .geometry import pack_chi_sparse, residue_level_names
from .constants import NUM_CHI, TARGETDIFF_RESIDUE_IDS
from pocketdiff.geometry.current_state import AMBIGUOUS_CHI_SLOTS


CACHE_FORMAT = "pocketdiff-v4-residue-cache"
CACHE_VERSION = 3


def _radius_edges(source: torch.Tensor, target: torch.Tensor, cutoff: float,
                  *, exclude_diagonal: bool = False) -> torch.Tensor:
    distance = torch.cdist(source.float(), target.float())
    keep = distance <= cutoff
    if exclude_diagonal:
        keep.fill_diagonal_(False)
    return torch.stack(torch.where(keep), dim=0).long().contiguous()


def source_fingerprint(path: str) -> Dict[str, object]:
    file = Path(path)
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(file.resolve()), "size": file.stat().st_size, "sha256": digest.hexdigest()}


def _torch_load_trusted(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def encode_complex(value, split: str) -> Dict[str, object]:
    apo = value.protein_pos_apo.float().contiguous()
    holo = value.protein_pos_holo.float().contiguous()
    ligand = value.ligand_pos_ref.float().contiguous()
    atom_to_residue = value.atom_to_residue.long().contiguous()
    nr = value.num_residues
    frame_index = torch.full((nr, 3), -1, dtype=torch.long)
    for atom, (name, residue) in enumerate(zip(value.protein_atom_name, atom_to_residue.tolist())):
        if name in ("N", "CA", "C"):
            slot = ("N", "CA", "C").index(name)
            if frame_index[residue, slot] < 0:
                frame_index[residue, slot] = atom
    residue_names = residue_level_names(
        value.protein_atom_name,
        value.protein_residue_name,
        atom_to_residue,
        nr,
    )
    chi_axis, chi_ptr, chi_downstream, topology_chi_mask, chi_quartet = pack_chi_sparse(
        value.protein_atom_name, residue_names, atom_to_residue
    )
    residue_type = torch.tensor(
        [TARGETDIFF_RESIDUE_IDS[name] for name in residue_names], dtype=torch.long
    )
    chi_apo = value.chi_apo.float().contiguous()[..., :NUM_CHI]
    chi_holo = value.chi_holo.float().contiguous()[..., :NUM_CHI]
    chi_supervision_mask = value.chi_mask.bool()[..., :NUM_CHI] & topology_chi_mask
    chi_ambiguous_mask = torch.zeros_like(topology_chi_mask)
    for residue_id, residue_name in enumerate(residue_names):
        slot = AMBIGUOUS_CHI_SLOTS.get(residue_name.upper())
        if slot is not None and slot < NUM_CHI:
            chi_ambiguous_mask[residue_id, slot] = True
    residue_center_apo = torch.zeros(nr, 3)
    residue_counts = torch.bincount(atom_to_residue, minlength=nr).float().clamp_min(1)
    residue_center_apo.index_add_(0, atom_to_residue, apo)
    residue_center_apo /= residue_counts[:, None]
    residue_feature = torch.zeros(nr, value.protein_feature.shape[-1])
    residue_feature.index_add_(0, atom_to_residue, value.protein_feature.float())
    residue_feature /= residue_counts[:, None]
    return {
        "sample_id": value.sample_id,
        "split": split,
        "input": {
            "apo_pos": apo,
            "protein_feature": value.protein_feature.float().contiguous(),
            "ligand_pos": ligand,
            "ligand_type": value.ligand_type_ref.long().contiguous(),
            "atom_to_residue": atom_to_residue,
            "residue_type": residue_type,
            "residue_feature": residue_feature,
            "residue_center_apo": residue_center_apo,
            "frame_index": frame_index,
            "chi_geometry_mask": topology_chi_mask,
            "chi_axis": chi_axis,
            "chi_ptr": chi_ptr,
            "chi_downstream": chi_downstream,
            "chi_quartet": chi_quartet,
            "chi_apo": chi_apo,
            "chi_ambiguous_mask": chi_ambiguous_mask,
            "rr_edge_index": _radius_edges(
                residue_center_apo, residue_center_apo, 12.0, exclude_diagonal=True
            ),
            "lr_edge_index": _radius_edges(ligand, residue_center_apo, 12.0),
            "ll_edge_index": _radius_edges(ligand, ligand, 5.0, exclude_diagonal=True),
            "atom_names": list(value.protein_atom_name),
            "residue_names": residue_names,
            "ligand_center": ligand.mean(0),
        },
        "target": {
            "holo_pos": holo,
            "chi_holo": chi_holo,
            "chi_supervision_mask": chi_supervision_mask,
        },
    }


def build_cache(source_path: str, output_path: str):
    source = _torch_load_trusted(source_path)
    if source.get("format") != "pocketdiff-medium-cache-v1":
        raise ValueError("unexpected Apo2Mol source cache format")
    splits = {
        split: [encode_complex(value, split) for value in values]
        for split, values in source["values"].items()
    }
    ids = {split: [sample["sample_id"] for sample in rows] for split, rows in splits.items()}
    if any(len(set(ids[a]) & set(ids[b])) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("source split identities overlap")
    payload = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "source": source_fingerprint(source_path),
        "geometry": "N-CA-C local frame; sparse chi topology; cached apo residue and ligand-residue radius graphs",
        "label_isolation": "model inputs and supervised targets are stored under separate input/target keys",
        "counts": {key: len(rows) for key, rows in splits.items()},
        "sample_ids": ids,
        "splits": splits,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "source": payload["source"],
        "counts": payload["counts"],
        "sample_ids": ids,
        "input_schema": sorted(splits["train"][0]["input"].keys()),
        "target_schema": sorted(splits["train"][0]["target"].keys()),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return payload


def load_cache(path: str):
    payload = _torch_load_trusted(path)
    if payload.get("format") != CACHE_FORMAT or payload.get("version") != CACHE_VERSION:
        raise ValueError("unsupported PocketDiff v4 cache")
    for split in ("train", "valid", "test"):
        if not payload["splits"].get(split):
            raise ValueError("missing or empty split: " + split)
        actual = [sample["sample_id"] for sample in payload["splits"][split]]
        if actual != payload["sample_ids"][split]:
            raise ValueError("sample identity manifest mismatch: " + split)
    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build CPU-resident PocketDiff v4 graph cache.")
    parser.add_argument(
        "--source", default="pocketdiff_v1/data/apo2mol_3000_cache.pt"
    )
    parser.add_argument(
        "--output", default="pocketdiff_v4/data/residue_graphs.pt"
    )
    args = parser.parse_args()
    payload = build_cache(args.source, args.output)
    print(json.dumps({
        "format": payload["format"],
        "counts": payload["counts"],
        "source_sha256": payload["source"]["sha256"],
        "manifest": str(Path(args.output).with_suffix(".json")),
    }, indent=2))


if __name__ == "__main__":
    main()
