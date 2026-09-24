"""Typed tensor contracts used by the new PocketDiff implementation.

This module deliberately contains no PDB parser, graph builder, model, or
TargetDiff import.  It is the first implementation boundary: later modules
must produce these objects and can rely on their shape/range checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


Tensor = torch.Tensor


def _check_tensor(
    name: str,
    value: Tensor,
    *,
    ndim: int,
    last_shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch.dtype] = None,
    floating: bool = False,
    finite: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(value.shape)}")
    if last_shape is not None and tuple(value.shape[-len(last_shape):]) != tuple(last_shape):
        raise ValueError(
            f"{name} must end with shape {tuple(last_shape)}, got {tuple(value.shape)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if floating and not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, got {value.dtype}")
    if finite and value.numel() and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _check_string_sequence(name: str, values: Sequence[str], length: int) -> None:
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} entries, got {len(values)}")
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{name} must contain only strings")


def _check_residue_ids(name: str, value: Tensor, num_residues: int) -> None:
    _check_tensor(name, value, ndim=1, dtype=torch.long)
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= num_residues):
        raise ValueError(f"{name} contains an out-of-range residue id")
    if num_residues and value.numel():
        present = torch.zeros(num_residues, dtype=torch.bool, device=value.device)
        present[value] = True
        if not bool(present.all()):
            missing = torch.where(~present)[0].tolist()
            raise ValueError(f"{name} does not reference every residue: missing={missing}")


def _check_categories(name: str, value: Tensor, upper_bound: int) -> None:
    _check_tensor(name, value, ndim=1, dtype=torch.long)
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= upper_bound):
        raise ValueError(
            f"{name} must contain categories in [0, {upper_bound - 1}], "
            f"got min={int(value.min())}, max={int(value.max())}"
        )


@dataclass
class PocketComplex:
    """One canonical apo/holo/ligand complex in the apo coordinate frame."""

    sample_id: str

    protein_pos_apo: Tensor
    protein_pos_holo: Tensor
    protein_feature: Tensor
    protein_element: Tensor
    protein_atom_name: List[str]
    protein_residue_name: List[str]
    atom_to_residue: Tensor
    residue_type: Tensor
    residue_chain_id: List[str]
    residue_sequence_id: List[str]
    frame_valid: Tensor
    chi_apo: Tensor
    chi_holo: Tensor
    chi_mask: Tensor

    ligand_pos_ref: Tensor
    ligand_type_ref: Tensor
    center_offset: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        _check_tensor("protein_pos_apo", self.protein_pos_apo, ndim=2, last_shape=(3,), floating=True, finite=True)
        _check_tensor("protein_pos_holo", self.protein_pos_holo, ndim=2, last_shape=(3,), floating=True, finite=True)
        if self.protein_pos_apo.shape != self.protein_pos_holo.shape:
            raise ValueError("protein_pos_apo and protein_pos_holo must have identical shapes")
        num_atoms = int(self.protein_pos_apo.shape[0])
        if num_atoms == 0:
            raise ValueError("a PocketComplex must contain at least one protein atom")

        _check_tensor("protein_feature", self.protein_feature, ndim=2, last_shape=(27,), floating=True, finite=True)
        if self.protein_feature.shape[0] != num_atoms:
            raise ValueError("protein_feature and protein coordinates have different atom counts")
        _check_tensor("protein_element", self.protein_element, ndim=1, dtype=torch.long)
        if self.protein_element.numel() != num_atoms:
            raise ValueError("protein_element and protein coordinates have different atom counts")
        _check_string_sequence("protein_atom_name", self.protein_atom_name, num_atoms)
        _check_string_sequence("protein_residue_name", self.protein_residue_name, num_atoms)

        _check_tensor("atom_to_residue", self.atom_to_residue, ndim=1, dtype=torch.long)
        if self.atom_to_residue.numel() != num_atoms:
            raise ValueError("atom_to_residue and protein coordinates have different atom counts")
        num_residues = len(self.residue_chain_id)
        if num_residues == 0:
            raise ValueError("a PocketComplex must contain at least one residue")
        _check_residue_ids("atom_to_residue", self.atom_to_residue, num_residues)
        _check_categories("residue_type", self.residue_type, 20)
        if self.residue_type.numel() != num_residues:
            raise ValueError("residue_type and residue metadata have different residue counts")
        _check_string_sequence("residue_sequence_id", self.residue_sequence_id, num_residues)
        _check_string_sequence("residue_chain_id", self.residue_chain_id, num_residues)

        _check_tensor("frame_valid", self.frame_valid, ndim=1, dtype=torch.bool)
        _check_tensor("chi_apo", self.chi_apo, ndim=2, last_shape=(5,), floating=True, finite=True)
        _check_tensor("chi_holo", self.chi_holo, ndim=2, last_shape=(5,), floating=True, finite=True)
        _check_tensor("chi_mask", self.chi_mask, ndim=2, last_shape=(5,), dtype=torch.bool)
        for name, value in (("frame_valid", self.frame_valid), ("chi_apo", self.chi_apo),
                            ("chi_holo", self.chi_holo), ("chi_mask", self.chi_mask)):
            if value.shape[0] != num_residues:
                raise ValueError(f"{name} and residue metadata have different residue counts")

        _check_tensor("ligand_pos_ref", self.ligand_pos_ref, ndim=2, last_shape=(3,), floating=True, finite=True)
        _check_tensor("ligand_type_ref", self.ligand_type_ref, ndim=1, dtype=torch.long)
        if self.ligand_type_ref.numel() != self.ligand_pos_ref.shape[0]:
            raise ValueError("ligand_type_ref and ligand_pos_ref have different atom counts")
        _check_categories("ligand_type_ref", self.ligand_type_ref, 13)
        _check_tensor("center_offset", self.center_offset, ndim=1, last_shape=(3,), floating=True, finite=True)

    @property
    def num_protein_atoms(self) -> int:
        return int(self.protein_pos_apo.shape[0])

    @property
    def num_residues(self) -> int:
        return len(self.residue_chain_id)

    @property
    def num_ligand_atoms(self) -> int:
        return int(self.ligand_pos_ref.shape[0])


@dataclass
class ResidueMetadata:
    """CPU-side metadata used by geometry and collate code."""

    protein_atom_name: List[List[str]]
    protein_residue_name: List[List[str]]
    residue_type: Tensor
    atom_to_residue_global: Tensor
    batch_residue: Tensor
    frame_valid_reference: Tensor
    chi_mask: Tensor
    chain_id: List[List[str]]
    residue_sequence_id: List[List[str]]

    def __post_init__(self) -> None:
        _check_tensor("residue_type", self.residue_type, ndim=1, dtype=torch.long)
        _check_categories("residue_type", self.residue_type, 20)
        num_residues = int(self.residue_type.numel())
        _check_tensor("atom_to_residue_global", self.atom_to_residue_global, ndim=1, dtype=torch.long)
        _check_residue_ids("atom_to_residue_global", self.atom_to_residue_global, num_residues)
        _check_tensor("batch_residue", self.batch_residue, ndim=1, dtype=torch.long)
        if self.batch_residue.numel() != num_residues:
            raise ValueError("batch_residue and residue_type have different lengths")
        _check_tensor("frame_valid_reference", self.frame_valid_reference, ndim=1, dtype=torch.bool)
        _check_tensor("chi_mask", self.chi_mask, ndim=2, last_shape=(5,), dtype=torch.bool)
        if self.frame_valid_reference.numel() != num_residues or self.chi_mask.shape[0] != num_residues:
            raise ValueError("residue metadata tensors have different residue counts")
        if len(self.chain_id) != len(self.residue_sequence_id):
            raise ValueError("chain_id and residue_sequence_id must have the same graph count")
        if len(self.protein_atom_name) != len(self.protein_residue_name):
            raise ValueError("protein atom/residue name metadata must have the same graph count")


@dataclass
class PocketBatchState:
    """Mutable coupled-sampler state; only ``protein_pos`` may be updated."""

    protein_pos: Tensor
    apo_pos_ref: Tensor
    protein_feature: Tensor
    atom_to_residue_local: Tensor
    atom_to_residue_global: Tensor
    batch_protein: Tensor
    batch_residue: Tensor
    ligand_pos: Tensor
    ligand_v: Tensor
    batch_ligand: Tensor
    k: Tensor
    targetdiff_t: Tensor
    pocket_s: Tensor
    center_offset: Tensor

    def __post_init__(self) -> None:
        _check_tensor("protein_pos", self.protein_pos, ndim=2, last_shape=(3,), floating=True, finite=True)
        _check_tensor("apo_pos_ref", self.apo_pos_ref, ndim=2, last_shape=(3,), floating=True, finite=True)
        if self.protein_pos.shape != self.apo_pos_ref.shape:
            raise ValueError("protein_pos and apo_pos_ref must have identical shapes")
        num_atoms = int(self.protein_pos.shape[0])
        _check_tensor("protein_feature", self.protein_feature, ndim=2, last_shape=(27,), floating=True, finite=True)
        if self.protein_feature.shape[0] != num_atoms:
            raise ValueError("protein_feature and protein_pos have different atom counts")
        _check_tensor("batch_protein", self.batch_protein, ndim=1, dtype=torch.long)
        if self.batch_protein.numel() != num_atoms or num_atoms == 0:
            raise ValueError("batch_protein must contain one graph id per protein atom")
        if int(self.batch_protein.min()) < 0:
            raise ValueError("batch_protein cannot contain negative graph ids")
        graph_count = int(self.batch_protein.max()) + 1
        _check_tensor("atom_to_residue_local", self.atom_to_residue_local, ndim=1, dtype=torch.long)
        _check_tensor("atom_to_residue_global", self.atom_to_residue_global, ndim=1, dtype=torch.long)
        if self.atom_to_residue_local.numel() != num_atoms or self.atom_to_residue_global.numel() != num_atoms:
            raise ValueError("residue mappings and protein_pos have different atom counts")
        _check_tensor("batch_residue", self.batch_residue, ndim=1, dtype=torch.long)
        if self.batch_residue.numel() == 0 or int(self.batch_residue.max()) + 1 != graph_count:
            raise ValueError("batch_residue must cover exactly the protein graphs")
        if self.atom_to_residue_global.numel() and int(self.atom_to_residue_global.max()) >= self.batch_residue.numel():
            raise ValueError("atom_to_residue_global references a missing residue")
        if not torch.equal(self.batch_residue[self.atom_to_residue_global], self.batch_protein):
            raise ValueError("atom_to_residue_global crosses graph boundaries")

        _check_tensor("ligand_pos", self.ligand_pos, ndim=2, last_shape=(3,), floating=True, finite=True)
        _check_tensor("ligand_v", self.ligand_v, ndim=1, dtype=torch.long)
        _check_tensor("batch_ligand", self.batch_ligand, ndim=1, dtype=torch.long)
        if self.ligand_v.numel() != self.ligand_pos.shape[0] or self.batch_ligand.numel() != self.ligand_pos.shape[0]:
            raise ValueError("ligand tensors have different atom counts")
        _check_categories("ligand_v", self.ligand_v, 13)
        if self.batch_ligand.numel() and (int(self.batch_ligand.min()) < 0 or int(self.batch_ligand.max()) >= graph_count):
            raise ValueError("batch_ligand contains an out-of-range graph id")

        for name, value in (("k", self.k), ("targetdiff_t", self.targetdiff_t)):
            _check_tensor(name, value, ndim=1, dtype=torch.long)
            if value.numel() != graph_count:
                raise ValueError(f"{name} must contain one value per graph")
        if self.k.numel() and (int(self.k.min()) < 0 or int(self.k.max()) > 19):
            raise ValueError("k must lie in [0, 19]")
        if self.targetdiff_t.numel() and (int(self.targetdiff_t.min()) < 0 or int(self.targetdiff_t.max()) > 999):
            raise ValueError("targetdiff_t must lie in [0, 999]")
        _check_tensor("pocket_s", self.pocket_s, ndim=1, floating=True, finite=True)
        if self.pocket_s.numel() != graph_count or (self.pocket_s.numel() and (self.pocket_s.min() < 0 or self.pocket_s.max() > 1)):
            raise ValueError("pocket_s must contain one value per graph in [0, 1]")
        _check_tensor("center_offset", self.center_offset, ndim=2, last_shape=(3,), floating=True, finite=True)
        if self.center_offset.shape[0] != graph_count:
            raise ValueError("center_offset must have shape [num_graphs, 3]")


@dataclass
class PocketDiffPrediction:
    """Pure model output; it must not mutate the input state."""

    remaining_translation_local: Tensor
    remaining_rotvec_local: Tensor
    remaining_chi: Optional[Tensor]
    frame_valid: Tensor
    diagnostics: Dict[str, Tensor]

    def __post_init__(self) -> None:
        _check_tensor("remaining_translation_local", self.remaining_translation_local, ndim=2, last_shape=(3,), floating=True, finite=True)
        _check_tensor("remaining_rotvec_local", self.remaining_rotvec_local, ndim=2, last_shape=(3,), floating=True, finite=True)
        if self.remaining_translation_local.shape != self.remaining_rotvec_local.shape:
            raise ValueError("translation and rotvec predictions must have identical shapes")
        num_residues = int(self.remaining_translation_local.shape[0])
        _check_tensor("frame_valid", self.frame_valid, ndim=1, dtype=torch.bool)
        if self.frame_valid.numel() != num_residues:
            raise ValueError("frame_valid and residue predictions have different lengths")
        if self.remaining_chi is not None:
            _check_tensor("remaining_chi", self.remaining_chi, ndim=2, last_shape=(5,), floating=True, finite=True)
            if self.remaining_chi.shape[0] != num_residues:
                raise ValueError("remaining_chi and residue predictions have different lengths")


@dataclass
class PocketStepOutput:
    """Result of applying one geometry solver update."""

    protein_pos_next: Tensor
    prediction: PocketDiffPrediction
    applied_translation_local: Tensor
    applied_rotvec_local: Tensor
    applied_chi: Optional[Tensor]
    diagnostics: Dict[str, Tensor]

    def __post_init__(self) -> None:
        _check_tensor("protein_pos_next", self.protein_pos_next, ndim=2, last_shape=(3,), floating=True, finite=True)
        _check_tensor("applied_translation_local", self.applied_translation_local, ndim=2, last_shape=(3,), floating=True, finite=True)
        _check_tensor("applied_rotvec_local", self.applied_rotvec_local, ndim=2, last_shape=(3,), floating=True, finite=True)
        if self.applied_translation_local.shape != self.applied_rotvec_local.shape:
            raise ValueError("applied translation and rotvec must have identical shapes")
        if self.applied_translation_local.shape[0] != self.prediction.frame_valid.numel():
            raise ValueError("applied residue updates and prediction have different lengths")
        if self.applied_chi is not None:
            _check_tensor("applied_chi", self.applied_chi, ndim=2, last_shape=(5,), floating=True, finite=True)
            if self.applied_chi.shape[0] != self.applied_translation_local.shape[0]:
                raise ValueError("applied_chi and residue updates have different lengths")


__all__ = [
    "PocketBatchState",
    "PocketComplex",
    "PocketDiffPrediction",
    "PocketStepOutput",
    "ResidueMetadata",
]
