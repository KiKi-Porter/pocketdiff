"""Minimal clean overfit trainer and checkpoint helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pocketdiff.models import PocketDiffModel

from .clean import CleanBatch, MotionLoss, masked_remaining_motion_loss


@dataclass(frozen=True)
class TrainRecord:
    step: int
    loss: float
    translation_loss: float
    rotation_loss: float


def train_clean_batch(
    model: PocketDiffModel,
    batch: CleanBatch,
    *,
    steps: int = 200,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    log_every: int = 10,
    max_grad_norm: Optional[float] = 10.0,
) -> List[TrainRecord]:
    if steps <= 0 or learning_rate <= 0.0 or log_every <= 0:
        raise ValueError("steps, learning_rate and log_every must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    records: List[TrainRecord] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**batch.model_kwargs())
        motion_loss = masked_remaining_motion_loss(
            prediction,
            batch.target_translation_local,
            batch.target_rotvec_local,
            batch.frame_valid,
        )
        if not torch.isfinite(motion_loss.loss):
            raise FloatingPointError(f"non-finite clean loss at step {step}")
        motion_loss.loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % log_every == 0:
            records.append(
                TrainRecord(
                    step=step + 1,
                    loss=float(motion_loss.loss.detach()),
                    translation_loss=float(motion_loss.translation_loss.detach()),
                    rotation_loss=float(motion_loss.rotation_loss.detach()),
                )
            )
    return records


def save_checkpoint(
    path: Path,
    model: PocketDiffModel,
    *,
    config: Dict[str, object],
    sample_ids: List[str],
    final_record: TrainRecord,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pocketdiff-clean-mvp-v1",
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "sample_ids": sample_ids,
            "final_record": final_record.__dict__,
        },
        path,
    )


__all__ = ["TrainRecord", "save_checkpoint", "train_clean_batch"]
