"""Compact dataset for frozen-PocketDiff bridge protein states."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets.pl_data import ProteinLigandData


TARGETDIFF_INPUT_FIELDS = (
    "protein_element",
    "protein_pos",
    "protein_atom_feature",
    "ligand_element",
    "ligand_pos",
    "ligand_atom_feature_full",
)


class BridgeStateDataset(Dataset):
    """Expose each saved bridge state as an independent TargetDiff graph.

    Static protein/ligand features are stored once per complex. Only the
    protein coordinate tensor is selected per state, avoiding 20-fold feature
    duplication in the on-disk cache.
    """

    @classmethod
    def from_payload(cls, payload, split: str):
        """Create a split view without deserializing the cache again."""
        dataset = cls.__new__(cls)
        Dataset.__init__(dataset)
        dataset._configure(payload, split)
        return dataset

    def __init__(self, cache_path: str | Path, split: str):
        payload = torch.load(
            str(cache_path), map_location="cpu", weights_only=False
        )
        self._configure(payload, split)

    def _configure(self, payload, split: str):
        if payload.get("metadata", {}).get("cache_version") != "targetdiff_bridge_finetune_v1":
            raise ValueError("Unsupported bridge fine-tuning cache payload")
        self.metadata = payload["metadata"]
        self.complexes = payload["complexes"]
        split_indices = payload.get("complex_indices_by_split", {})
        if split not in split_indices:
            raise KeyError(f"Unknown split {split!r}; available={list(split_indices)}")
        self.complex_indices = [int(index) for index in split_indices[split]]
        self.mapping = [
            (complex_index, state_index)
            for complex_index in self.complex_indices
            for state_index in range(int(self.complexes[complex_index]["protein_pos_states"].shape[0]))
        ]

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> ProteinLigandData:
        complex_index, state_index = self.mapping[index]
        record = self.complexes[complex_index]
        sample = dict(record["static"])
        sample["protein_pos"] = record["protein_pos_states"][state_index]
        sample["num_nodes"] = int(sample["protein_pos"].shape[0] + sample["ligand_pos"].shape[0])
        data = ProteinLigandData(**sample)
        data.id = int(complex_index)
        return data


class TargetDiffInputDataset(Dataset):
    """Project heterogeneous source records onto TargetDiff's loss inputs."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ProteinLigandData:
        sample = self.dataset[index]
        fields = {name: getattr(sample, name) for name in TARGETDIFF_INPUT_FIELDS}
        return ProteinLigandData(**fields)
