from __future__ import annotations

from typing import Dict, List

import torch

from .constants import NUM_CHI


def collate_complexes(samples: List[Dict[str, object]]) -> Dict[str, object]:
    """Concatenate variable-size complexes while offsetting every graph index."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")

    atoms = residues = ligands = downstream_atoms = 0
    input_parts: Dict[str, list] = {
        key: [] for key in (
            "apo_pos", "protein_feature", "ligand_pos", "ligand_type",
            "atom_to_residue", "residue_type", "residue_feature",
            "residue_center_apo", "frame_index", "backbone_mask", "chi_geometry_mask",
            "chi_axis", "chi_ptr", "chi_downstream", "chi_quartet",
            "chi_apo", "chi_ambiguous_mask", "rr_edge_index",
            "lr_edge_index", "ll_edge_index",
        )
    }
    target_parts = {
        "holo_pos": [],
        "chi_holo": [],
        "chi_supervision_mask": [],
    }
    atom_ptr, residue_ptr, ligand_ptr = [0], [0], [0]
    sample_ids = []

    for sample in samples:
        x, y = sample["input"], sample["target"]
        num_atoms = x["apo_pos"].shape[0]
        num_residues = x["residue_type"].shape[0]
        num_ligands = x["ligand_type"].shape[0]

        input_parts["apo_pos"].append(x["apo_pos"])
        input_parts["protein_feature"].append(x["protein_feature"])
        input_parts["backbone_mask"].append(
            x.get(
                "backbone_mask",
                x["protein_feature"][:, -1].to(dtype=torch.bool),
            )
        )
        input_parts["ligand_pos"].append(x["ligand_pos"])
        input_parts["ligand_type"].append(x["ligand_type"])
        input_parts["atom_to_residue"].append(x["atom_to_residue"] + residues)
        input_parts["residue_type"].append(x["residue_type"])
        input_parts["residue_feature"].append(x["residue_feature"])
        input_parts["residue_center_apo"].append(x["residue_center_apo"])
        frame_index = x["frame_index"]
        input_parts["frame_index"].append(
            torch.where(frame_index >= 0, frame_index + atoms, frame_index)
        )
        input_parts["chi_geometry_mask"].append(x["chi_geometry_mask"])
        chi_axis = x["chi_axis"]
        input_parts["chi_axis"].append(
            torch.where(chi_axis >= 0, chi_axis + atoms, chi_axis)
        )
        chi_quartet = x.get(
            "chi_quartet",
            torch.full(
                (num_residues, NUM_CHI, 4), -1, dtype=torch.long
            ),
        )
        input_parts["chi_quartet"].append(
            torch.where(chi_quartet >= 0, chi_quartet + atoms, chi_quartet)
        )
        input_parts["chi_apo"].append(
            x.get("chi_apo", torch.zeros((num_residues, NUM_CHI)))
        )
        input_parts["chi_ambiguous_mask"].append(
            x.get(
                "chi_ambiguous_mask",
                torch.zeros((num_residues, NUM_CHI), dtype=torch.bool),
            )
        )
        input_parts["chi_ptr"].append(x["chi_ptr"][1:] + downstream_atoms)
        input_parts["chi_downstream"].append(x["chi_downstream"] + atoms)
        input_parts["rr_edge_index"].append(x["rr_edge_index"] + residues)
        lr_edges = x["lr_edge_index"].clone()
        if lr_edges.numel():
            lr_edges[0] += ligands
            lr_edges[1] += residues
        input_parts["lr_edge_index"].append(lr_edges)
        input_parts["ll_edge_index"].append(x["ll_edge_index"] + ligands)

        for key in target_parts:
            target_parts[key].append(y[key])
        sample_ids.append(sample["sample_id"])

        atoms += num_atoms
        residues += num_residues
        ligands += num_ligands
        downstream_atoms += x["chi_downstream"].numel()
        atom_ptr.append(atoms)
        residue_ptr.append(residues)
        ligand_ptr.append(ligands)

    input_batch = {
        key: torch.cat(parts, dim=0) if key not in ("chi_ptr", "rr_edge_index", "lr_edge_index", "ll_edge_index", "frame_index", "chi_axis")
        else torch.cat(parts, dim=1) if key in ("rr_edge_index", "lr_edge_index", "ll_edge_index")
        else torch.cat(parts, dim=0)
        for key, parts in input_parts.items()
    }
    # CSR pointers use a shared zero entry followed by each sample's shifted tail.
    input_batch["chi_ptr"] = torch.cat(
        [torch.zeros(1, dtype=torch.long), *input_parts["chi_ptr"]]
    )
    input_batch["atom_ptr"] = torch.tensor(atom_ptr, dtype=torch.long)
    input_batch["residue_ptr"] = torch.tensor(residue_ptr, dtype=torch.long)
    input_batch["ligand_ptr"] = torch.tensor(ligand_ptr, dtype=torch.long)
    target_batch = {
        key: torch.cat(parts, dim=0) for key, parts in target_parts.items()
    }
    target_batch["sample_id"] = sample_ids
    return {"input": input_batch, "target": target_batch}
