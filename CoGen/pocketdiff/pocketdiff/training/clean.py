"""Clean teacher-forced examples and collate for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence

import torch

from pocketdiff.data.schema import PocketComplex, PocketDiffPrediction
from pocketdiff.geometry.bridge import remaining_transform_current_to_holo
from pocketdiff.geometry.chi import periodic_chi_delta
from pocketdiff.geometry.frames import build_residue_frames


@dataclass(frozen=True)
class CleanExample:
    complex_value: PocketComplex
    target_translation_local: torch.Tensor
    target_rotvec_local: torch.Tensor
    target_valid: torch.Tensor


@dataclass(frozen=True)
class CleanBatch:
    protein_pos: torch.Tensor
    protein_pos_holo: torch.Tensor
    apo_pos_ref: torch.Tensor
    protein_feature: torch.Tensor
    atom_to_residue_global: torch.Tensor
    residue_type: torch.Tensor
    frame_valid: torch.Tensor
    chi_apo: torch.Tensor
    chi_holo: torch.Tensor
    chi_mask: torch.Tensor
    batch_protein: torch.Tensor
    batch_residue: torch.Tensor
    ligand_pos: torch.Tensor
    ligand_v: torch.Tensor
    batch_ligand: torch.Tensor
    targetdiff_t: torch.Tensor
    pocket_k: torch.Tensor
    protein_atom_name: List[str]
    target_translation_local: torch.Tensor
    target_rotvec_local: torch.Tensor
    sample_ids: List[str]

    def model_kwargs(self) -> Dict[str, object]:
        return {
            "protein_pos": self.protein_pos,
            "apo_pos_ref": self.apo_pos_ref,
            "protein_feature": self.protein_feature,
            "atom_to_residue_global": self.atom_to_residue_global,
            "residue_type": self.residue_type,
            "frame_valid": self.frame_valid,
            # ``chi_holo`` is deliberately absent: it is a supervision-only
            # endpoint label and must never be visible to the model.
            "chi_apo": self.chi_apo,
            "chi_mask": self.chi_mask,
            "batch_protein": self.batch_protein,
            "batch_residue": self.batch_residue,
            "ligand_pos": self.ligand_pos,
            "ligand_v": self.ligand_v,
            "batch_ligand": self.batch_ligand,
            "targetdiff_t": self.targetdiff_t,
            "pocket_k": self.pocket_k,
            "protein_atom_name": self.protein_atom_name,
        }

    @property
    def target_chi(self) -> torch.Tensor:
        """Shortest periodic holo-minus-apo χ label in radians."""

        return periodic_chi_delta(self.chi_apo, self.chi_holo, self.chi_mask)


def make_clean_example(complex_value: PocketComplex) -> CleanExample:
    """Build the apo→holo remaining-transform label for one complex."""

    apo_frames = build_residue_frames(
        complex_value.protein_pos_apo,
        complex_value.atom_to_residue,
        complex_value.protein_atom_name,
        num_residues=complex_value.num_residues,
    )
    holo_frames = build_residue_frames(
        complex_value.protein_pos_holo,
        complex_value.atom_to_residue,
        complex_value.protein_atom_name,
        num_residues=complex_value.num_residues,
    )
    valid = complex_value.frame_valid & apo_frames.valid & holo_frames.valid
    remaining = remaining_transform_current_to_holo(
        apo_frames.origins,
        apo_frames.frames,
        holo_frames.origins,
        holo_frames.frames,
        frame_valid=valid,
    )
    return CleanExample(
        complex_value=complex_value,
        target_translation_local=remaining.translation_local,
        target_rotvec_local=remaining.rotvec_local,
        target_valid=valid,
    )


def collate_clean_examples(examples: Sequence[CleanExample]) -> CleanBatch:
    if not examples:
        raise ValueError("at least one clean example is required")
    protein_pos = []
    protein_pos_holo = []
    apo_pos_ref = []
    protein_feature = []
    atom_to_residue = []
    residue_type = []
    frame_valid = []
    chi_apo = []
    chi_holo = []
    chi_mask = []
    batch_protein = []
    batch_residue = []
    ligand_pos = []
    ligand_v = []
    batch_ligand = []
    target_translation = []
    target_rotvec = []
    sample_ids = []
    protein_atom_name: List[str] = []
    residue_offset = 0
    for graph_id, example in enumerate(examples):
        value = example.complex_value
        n_protein = value.num_protein_atoms
        n_residue = value.num_residues
        n_ligand = value.num_ligand_atoms
        protein_pos.append(value.protein_pos_apo)
        protein_pos_holo.append(value.protein_pos_holo)
        apo_pos_ref.append(value.protein_pos_apo)
        protein_feature.append(value.protein_feature)
        atom_to_residue.append(value.atom_to_residue + residue_offset)
        residue_type.append(value.residue_type)
        frame_valid.append(example.target_valid)
        chi_apo.append(value.chi_apo)
        chi_holo.append(value.chi_holo)
        chi_mask.append(value.chi_mask)
        batch_protein.append(torch.full((n_protein,), graph_id, dtype=torch.long))
        batch_residue.append(torch.full((n_residue,), graph_id, dtype=torch.long))
        ligand_pos.append(value.ligand_pos_ref)
        ligand_v.append(value.ligand_type_ref)
        batch_ligand.append(torch.full((n_ligand,), graph_id, dtype=torch.long))
        target_translation.append(example.target_translation_local)
        target_rotvec.append(example.target_rotvec_local)
        protein_atom_name.extend(value.protein_atom_name)
        sample_ids.append(value.sample_id)
        residue_offset += n_residue

    return CleanBatch(
        protein_pos=torch.cat(protein_pos, dim=0),
        protein_pos_holo=torch.cat(protein_pos_holo, dim=0),
        apo_pos_ref=torch.cat(apo_pos_ref, dim=0),
        protein_feature=torch.cat(protein_feature, dim=0),
        atom_to_residue_global=torch.cat(atom_to_residue, dim=0),
        residue_type=torch.cat(residue_type, dim=0),
        frame_valid=torch.cat(frame_valid, dim=0),
        chi_apo=torch.cat(chi_apo, dim=0),
        chi_holo=torch.cat(chi_holo, dim=0),
        chi_mask=torch.cat(chi_mask, dim=0),
        batch_protein=torch.cat(batch_protein, dim=0),
        batch_residue=torch.cat(batch_residue, dim=0),
        ligand_pos=torch.cat(ligand_pos, dim=0),
        ligand_v=torch.cat(ligand_v, dim=0),
        batch_ligand=torch.cat(batch_ligand, dim=0),
        targetdiff_t=torch.full((len(examples),), 199, dtype=torch.long),
        pocket_k=torch.zeros(len(examples), dtype=torch.long),
        protein_atom_name=protein_atom_name,
        target_translation_local=torch.cat(target_translation, dim=0),
        target_rotvec_local=torch.cat(target_rotvec, dim=0),
        sample_ids=sample_ids,
    )


@dataclass(frozen=True)
class MotionLoss:
    loss: torch.Tensor
    translation_loss: torch.Tensor
    rotation_loss: torch.Tensor
    valid_residue_count: int


@dataclass(frozen=True)
class ChiLoss:
    """Masked periodic χ regression loss."""

    loss: torch.Tensor
    valid_chi_count: int
    valid_residue_count: int


def masked_remaining_motion_loss(
    prediction: PocketDiffPrediction,
    target_translation_local: torch.Tensor,
    target_rotvec_local: torch.Tensor,
    target_valid: torch.Tensor,
) -> MotionLoss:
    if target_translation_local.shape != prediction.remaining_translation_local.shape:
        raise ValueError("target translation and prediction shapes differ")
    if target_rotvec_local.shape != prediction.remaining_rotvec_local.shape:
        raise ValueError("target rotvec and prediction shapes differ")
    if target_valid.dtype != torch.bool or target_valid.shape != prediction.frame_valid.shape:
        raise ValueError("target_valid must be BoolTensor [Nr]")
    valid = target_valid & prediction.frame_valid
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("masked motion loss has no valid residues")
    translation_error = (prediction.remaining_translation_local - target_translation_local).square().mean(dim=-1)
    rotation_error = (prediction.remaining_rotvec_local - target_rotvec_local).square().mean(dim=-1)
    translation_loss = translation_error[valid].mean()
    rotation_loss = rotation_error[valid].mean()
    return MotionLoss(
        loss=translation_loss + rotation_loss,
        translation_loss=translation_loss,
        rotation_loss=rotation_loss,
        valid_residue_count=count,
    )


def masked_periodic_chi_loss(
    prediction: PocketDiffPrediction,
    chi_apo: torch.Tensor,
    chi_holo: torch.Tensor,
    chi_mask: torch.Tensor,
) -> ChiLoss:
    """Regress current→holo χ deltas with the shortest periodic error.

    The endpoint angles are only used to construct the target.  A model output
    is interpreted as an actual remaining angle in radians; no bridge-rate
    normalization is applied here.  Residues with invalid rigid frames and
    individual χ entries with missing atoms are excluded independently.
    """

    if prediction.remaining_chi is None:
        raise ValueError("prediction does not contain remaining_chi")
    if chi_apo.ndim != 2 or chi_holo.shape != chi_apo.shape or chi_apo.shape[-1] != 5:
        raise ValueError("chi_apo and chi_holo must have identical shape [Nr, 5]")
    if chi_mask.dtype != torch.bool or chi_mask.shape != chi_apo.shape:
        raise ValueError("chi_mask must be BoolTensor with shape [Nr, 5]")
    if prediction.remaining_chi.shape != chi_apo.shape:
        raise ValueError("prediction and chi labels have different shapes")
    if not (torch.isfinite(chi_apo).all() and torch.isfinite(chi_holo).all()):
        raise ValueError("chi labels must be finite")
    target = periodic_chi_delta(chi_apo, chi_holo, chi_mask)
    valid = chi_mask & prediction.frame_valid[:, None]
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError("masked chi loss has no valid chi entries")
    periodic_error = torch.atan2(
        torch.sin(prediction.remaining_chi - target),
        torch.cos(prediction.remaining_chi - target),
    )
    chi_loss = periodic_error.square()[valid].mean()
    return ChiLoss(
        loss=chi_loss,
        valid_chi_count=valid_count,
        valid_residue_count=int(valid.any(dim=1).sum().item()),
    )


# The shorter name is useful in later joint trainers while retaining the
# explicit periodic name for callers that need to document the metric.
masked_chi_loss = masked_periodic_chi_loss


def masked_bridge_rate_loss(
    prediction: PocketDiffPrediction,
    target_translation_local: torch.Tensor,
    target_rotvec_local: torch.Tensor,
    target_valid: torch.Tensor,
    pocket_k: torch.Tensor,
    batch_residue: torch.Tensor,
) -> MotionLoss:
    """Supervise remaining/r, r=(20-k)/20, with equal per-residue rate weight.

    Prediction and targets enter in actual remaining units. This paired loss is
    intended for the bridge_rate model on teacher-forced bridge states.
    """
    if pocket_k.dtype != torch.long or pocket_k.ndim != 1 or pocket_k.numel() == 0:
        raise ValueError("pocket_k must be non-empty LongTensor [B]")
    if int(pocket_k.min()) < 0 or int(pocket_k.max()) > 19:
        raise ValueError("pocket_k must lie in [0, 19]")
    if batch_residue.dtype != torch.long or batch_residue.shape != prediction.frame_valid.shape:
        raise ValueError("batch_residue must be LongTensor [Nr]")
    if batch_residue.numel() and (int(batch_residue.min()) < 0 or int(batch_residue.max()) >= pocket_k.numel()):
        raise ValueError("batch_residue contains an out-of-range graph id")
    # Validate before scaling to prevent accidental broadcasting of malformed labels.
    if target_translation_local.shape != prediction.remaining_translation_local.shape:
        raise ValueError("target translation and prediction shapes differ")
    if target_rotvec_local.shape != prediction.remaining_rotvec_local.shape:
        raise ValueError("target rotvec and prediction shapes differ")
    fraction = (20 - pocket_k[batch_residue]).to(prediction.remaining_translation_local.dtype)[:, None] / 20
    normalized = replace(
        prediction,
        remaining_translation_local=prediction.remaining_translation_local / fraction,
        remaining_rotvec_local=prediction.remaining_rotvec_local / fraction,
    )
    return masked_remaining_motion_loss(normalized, target_translation_local / fraction,
                                        target_rotvec_local / fraction, target_valid)


__all__ = [
    "masked_bridge_rate_loss",
    "CleanBatch",
    "CleanExample",
    "MotionLoss",
    "ChiLoss",
    "collate_clean_examples",
    "make_clean_example",
    "masked_remaining_motion_loss",
    "masked_periodic_chi_loss",
    "masked_chi_loss",
]
