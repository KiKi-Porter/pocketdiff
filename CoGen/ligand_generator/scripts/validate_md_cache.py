from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from datasets.md_dataset import PreparedMDDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/md300/prepared_targetdiff_md300_10frames.pt"),
    )
    args = parser.parse_args()

    payload = torch.load(
        str(args.cache),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Prepared cache must be a dictionary payload")
    metadata = payload.get("metadata", {})
    samples = payload.get("samples", [])
    records = payload.get("records", [])
    if not samples or len(samples) != len(records):
        raise ValueError("Prepared cache has inconsistent samples/records")
    if metadata.get("feature_contract", {}).get("protein_atom_feature_dim") != 27:
        raise ValueError("Protein feature contract is not 27D")
    if metadata.get("feature_contract", {}).get("ligand_atom_class_dim") != 13:
        raise ValueError("Ligand feature contract is not 13 classes")

    for index, sample in enumerate(samples):
        protein_pos = sample["protein_pos"]
        ligand_pos = sample["ligand_pos"]
        protein_feature = sample["protein_atom_feature"]
        ligand_feature = sample["ligand_atom_feature_full"]
        bond_index = sample["ligand_bond_index"]
        bond_type = sample["ligand_bond_type"]
        if protein_pos.ndim != 2 or protein_pos.shape[-1] != 3:
            raise ValueError(f"Invalid protein coordinates at sample {index}")
        if ligand_pos.ndim != 2 or ligand_pos.shape[-1] != 3:
            raise ValueError(f"Invalid ligand coordinates at sample {index}")
        if protein_feature.shape != (protein_pos.shape[0], 27):
            raise ValueError(f"Invalid protein feature shape at sample {index}")
        if ligand_feature.shape != (ligand_pos.shape[0],):
            raise ValueError(f"Invalid ligand feature shape at sample {index}")
        if ligand_feature.min().item() < 0 or ligand_feature.max().item() >= 13:
            raise ValueError(f"Invalid ligand class at sample {index}")
        if bond_index.shape[0] != 2 or bond_index.shape[1] != bond_type.shape[0]:
            raise ValueError(f"Invalid ligand bond shape at sample {index}")
        if bond_index.numel():
            if bond_index.min().item() < 0 or bond_index.max().item() >= ligand_pos.shape[0]:
                raise ValueError(f"Invalid ligand bond index at sample {index}")
        if not (
            torch.isfinite(protein_pos).all()
            and torch.isfinite(ligand_pos).all()
            and torch.isfinite(protein_feature).all()
        ):
            raise ValueError(f"Non-finite tensor at sample {index}")

    dataset = PreparedMDDataset(args.cache, split="train")
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                follow_batch=[
                    "protein_element",
                    "ligand_element",
                    "ligand_bond_type",
                ],
                exclude_keys=["ligand_nbh_list"],
            )
        )
    )
    if batch.protein_atom_feature.shape[-1] != 27:
        raise ValueError("Batched protein feature dimension is not 27")
    if batch.ligand_atom_feature_full.min().item() < 0:
        raise ValueError("Batched ligand classes are invalid")

    result = {
        "passed": True,
        "cache": str(args.cache),
        "sample_count": len(samples),
        "split_counts": metadata.get("split_counts", {}),
        "protein_feature_dim": int(samples[0]["protein_atom_feature"].shape[-1]),
        "ligand_class_dim": 13,
        "train_batch_protein_shape": list(batch.protein_atom_feature.shape),
        "train_batch_ligand_shape": list(batch.ligand_atom_feature_full.shape),
        "rejection_reasons": metadata.get("rejection_reasons", {}),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
