"""Small clean endpoint trainer for the rigid and χ heads together."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List

import torch

from pocketdiff.models import PocketDiffModel

from .clean import (
    ChiLoss,
    CleanBatch,
    MotionLoss,
    masked_periodic_chi_loss,
    masked_remaining_motion_loss,
)


@dataclass(frozen=True)
class JointLoss:
    loss: torch.Tensor
    motion_loss: torch.Tensor
    chi_loss: torch.Tensor
    translation_loss: torch.Tensor
    rotation_loss: torch.Tensor
    valid_residue_count: int
    valid_chi_count: int


@dataclass(frozen=True)
class JointTrainRecord:
    step: int
    loss: float
    motion_loss: float
    chi_loss: float
    translation_loss: float
    rotation_loss: float
    valid_residue_count: int
    valid_chi_count: int
    gradient_norm_before_clip: float


def masked_joint_clean_loss(
    prediction,
    batch: CleanBatch,
    *,
    chi_weight: float = 1.0,
) -> JointLoss:
    """Combine physical rigid remaining loss with periodic χ loss."""

    if not math.isfinite(chi_weight) or chi_weight < 0.0:
        raise ValueError("chi_weight must be finite and non-negative")
    motion = masked_remaining_motion_loss(
        prediction,
        batch.target_translation_local,
        batch.target_rotvec_local,
        batch.frame_valid,
    )
    chi = masked_periodic_chi_loss(
        prediction, batch.chi_apo, batch.chi_holo, batch.chi_mask
    )
    total = motion.loss + float(chi_weight) * chi.loss
    if not torch.isfinite(total):
        raise FloatingPointError("joint clean loss is non-finite")
    return JointLoss(
        loss=total,
        motion_loss=motion.loss,
        chi_loss=chi.loss,
        translation_loss=motion.translation_loss,
        rotation_loss=motion.rotation_loss,
        valid_residue_count=motion.valid_residue_count,
        valid_chi_count=chi.valid_chi_count,
    )


def train_joint_clean_batch(
    model: PocketDiffModel,
    batch: CleanBatch,
    *,
    steps: int = 200,
    learning_rate: float = 1.0e-3,
    chi_weight: float = 1.0,
    log_every: int = 10,
    max_grad_norm: float = 10.0,
) -> List[JointTrainRecord]:
    """Train both heads on one fixed clean endpoint batch."""

    if not getattr(model, "predict_chi", False):
        raise ValueError("joint clean training requires model predict_chi=True")
    if steps <= 0 or learning_rate <= 0.0 or log_every <= 0 or max_grad_norm <= 0.0:
        raise ValueError("steps, learning_rate, log_every and max_grad_norm must be positive")
    if not math.isfinite(learning_rate) or not math.isfinite(max_grad_norm):
        raise ValueError("learning_rate and max_grad_norm must be finite")
    # Validate the data and the χ head before allocating an optimizer.
    model.eval()
    with torch.no_grad():
        initial_prediction = model(**batch.model_kwargs())
        masked_joint_clean_loss(initial_prediction, batch, chi_weight=chi_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    records: List[JointTrainRecord] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**batch.model_kwargs())
        loss = masked_joint_clean_loss(prediction, batch, chi_weight=chi_weight)
        loss.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("non-finite joint model parameter after optimizer step")
        if step == 0 or step == steps - 1 or (step + 1) % log_every == 0:
            records.append(
                JointTrainRecord(
                    step=step + 1,
                    loss=float(loss.loss.detach()),
                    motion_loss=float(loss.motion_loss.detach()),
                    chi_loss=float(loss.chi_loss.detach()),
                    translation_loss=float(loss.translation_loss.detach()),
                    rotation_loss=float(loss.rotation_loss.detach()),
                    valid_residue_count=loss.valid_residue_count,
                    valid_chi_count=loss.valid_chi_count,
                    gradient_norm_before_clip=float(norm),
                )
            )
    return records


@torch.no_grad()
def evaluate_joint_clean(
    model: PocketDiffModel,
    batch: CleanBatch,
    *,
    chi_weight: float = 1.0,
) -> JointLoss:
    """Evaluate the joint endpoint loss without changing model mode."""

    was_training = model.training
    model.eval()
    try:
        prediction = model(**batch.model_kwargs())
        return masked_joint_clean_loss(prediction, batch, chi_weight=chi_weight)
    finally:
        model.train(was_training)


def save_joint_checkpoint(
    path: Path,
    model: PocketDiffModel,
    *,
    config: Dict[str, object],
    sample_ids: List[str],
    final_record: JointTrainRecord,
) -> None:
    """Save an explicit, independent Phase 29 checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pocketdiff-joint-clean-v1",
            "model_state_dict": model.state_dict(),
            "model_config": dict(config),
            "sample_ids": list(sample_ids),
            "final_record": final_record.__dict__,
        },
        path,
    )


__all__ = [
    "JointLoss",
    "JointTrainRecord",
    "evaluate_joint_clean",
    "masked_joint_clean_loss",
    "save_joint_checkpoint",
    "train_joint_clean_batch",
]
