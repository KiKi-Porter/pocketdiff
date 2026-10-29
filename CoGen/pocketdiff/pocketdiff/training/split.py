"""Deterministic sample splitting for clean PocketDiff benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from .clean import CleanExample


@dataclass(frozen=True)
class CleanSplit:
    """Disjoint clean examples and the permutation used to obtain them."""

    train: List[CleanExample]
    holdout: List[CleanExample]
    permutation: List[int]

    @property
    def train_ids(self) -> List[str]:
        return [example.complex_value.sample_id for example in self.train]

    @property
    def holdout_ids(self) -> List[str]:
        return [example.complex_value.sample_id for example in self.holdout]


def deterministic_clean_split(
    examples: Sequence[CleanExample],
    *,
    holdout: int,
    seed: int,
) -> CleanSplit:
    """Split examples without replacement using a CPU generator and fixed seed.

    The returned order is deterministic and is written to experiment reports by
    the generalization smoke.  The input sequence is never modified.
    """

    total = len(examples)
    if total < 2:
        raise ValueError("at least two examples are required")
    if holdout <= 0 or holdout >= total:
        raise ValueError("holdout must satisfy 0 < holdout < len(examples)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation_tensor = torch.randperm(total, generator=generator, device="cpu")
    permutation = [int(index) for index in permutation_tensor.tolist()]
    holdout_indices = permutation[:holdout]
    train_indices = permutation[holdout:]
    return CleanSplit(
        train=[examples[index] for index in train_indices],
        holdout=[examples[index] for index in holdout_indices],
        permutation=permutation,
    )


__all__ = ["CleanSplit", "deterministic_clean_split"]
