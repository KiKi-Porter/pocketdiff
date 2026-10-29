from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets.pl_data import ProteinLigandData


class PreparedMDDataset(Dataset):
    """Dataset backed by the serialized native TargetDiff MD cache."""

    @classmethod
    def from_payload(cls, payload, split: str | None = None):
        """Create a split view without deserializing the cache again."""
        dataset = cls.__new__(cls)
        Dataset.__init__(dataset)
        dataset._configure(payload, split)
        return dataset

    def __init__(
        self,
        cache_path: str | Path,
        split: str | None = None,
    ):
        super().__init__()
        payload = torch.load(
            str(cache_path),
            map_location="cpu",
            weights_only=False,
        )
        self._configure(payload, split)

    def _configure(self, payload, split: str | None = None):
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError("Invalid prepared MD cache payload")
        self.metadata = payload.get("metadata", {})
        samples = payload["samples"]
        split_indices = payload.get("indices_by_split", {})
        if split is None:
            self.indices = list(range(len(samples)))
        else:
            if split not in split_indices:
                raise KeyError(f"Unknown split {split!r}; available={list(split_indices)}")
            self.indices = [int(index) for index in split_indices[split]]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> ProteinLigandData:
        sample = ProteinLigandData(**self.samples[self.indices[index]])
        sample.id = self.indices[index]
        return sample
