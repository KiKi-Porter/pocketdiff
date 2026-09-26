"""Independent scalar distance-invariant encoder for the PocketDiff MVP.

This is intentionally a small replacement point for the later TargetDiff
UniTransformer adapter.  It never consumes absolute coordinate directions:
messages depend on scalar node features and pairwise distances only, so a
global rigid transform leaves hidden states unchanged.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DistanceInvariantEncoder(nn.Module):
    def __init__(self, *, hidden_dim: int = 128, num_layers: int = 2, knn: int = 32, num_rbf: int = 16) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_layers <= 0 or knn <= 0 or num_rbf <= 0:
            raise ValueError("hidden_dim, num_layers, knn and num_rbf must be positive")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.knn = knn
        self.num_rbf = num_rbf
        self.protein_atom_emb = nn.Linear(27, hidden_dim)
        self.ligand_atom_emb = nn.Linear(13, hidden_dim)
        self.node_indicator = nn.Parameter(torch.zeros(2, hidden_dim))
        centers = torch.linspace(0.0, 10.0, num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_width = 10.0 / max(num_rbf - 1, 1)
        self.edge_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.update_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        )

    def _build_knn_edges(self, positions: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_parts = []
        dst_parts = []
        distance_parts = []
        graph_ids = torch.unique(batch, sorted=True)
        for graph_id in graph_ids.tolist():
            graph_indices = torch.where(batch == graph_id)[0]
            if graph_indices.numel() < 2:
                continue
            graph_pos = positions[graph_indices]
            distances = torch.cdist(graph_pos, graph_pos, p=2)
            distances = distances.clone()
            distances.fill_diagonal_(float("inf"))
            neighbors = min(self.knn, graph_indices.numel() - 1)
            nearest_distances, nearest_local = torch.topk(
                distances, k=neighbors, dim=1, largest=False, sorted=False
            )
            dst_local = torch.arange(
                graph_indices.numel(), device=positions.device, dtype=torch.long
            ).repeat_interleave(neighbors)
            src_local = nearest_local.reshape(-1)
            src_parts.append(graph_indices[src_local])
            dst_parts.append(graph_indices[dst_local])
            distance_parts.append(nearest_distances.reshape(-1))
        if not src_parts:
            empty_long = torch.empty(0, dtype=torch.long, device=positions.device)
            empty_float = torch.empty(0, dtype=positions.dtype, device=positions.device)
            return empty_long, empty_long, empty_float
        return torch.cat(src_parts), torch.cat(dst_parts), torch.cat(distance_parts)

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.numel() == 0:
            return distances.new_empty((0, self.num_rbf))
        centers = self.rbf_centers.to(dtype=distances.dtype)
        return torch.exp(-((distances[:, None] - centers[None, :]) / self.rbf_width).square())

    def forward(
        self,
        protein_pos: torch.Tensor,
        protein_feature: torch.Tensor,
        batch_protein: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        protein_time: torch.Tensor,
        ligand_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if protein_feature.ndim != 2 or protein_feature.shape[-1] != 27:
            raise ValueError("protein_feature must have shape [Np, 27]")
        if ligand_v.ndim != 1 or ligand_v.dtype != torch.long or ligand_v.shape[0] != ligand_pos.shape[0]:
            raise ValueError("ligand_v must be LongTensor with one value per ligand atom")
        protein_h = self.protein_atom_emb(protein_feature.to(dtype=torch.float32))
        ligand_one_hot = F.one_hot(ligand_v, num_classes=13).to(dtype=torch.float32)
        ligand_h = self.ligand_atom_emb(ligand_one_hot)
        protein_h = protein_h + self.node_indicator[0]
        ligand_h = ligand_h + self.node_indicator[1]
        protein_h = protein_h + protein_time[batch_protein]
        ligand_h = ligand_h + ligand_time[batch_ligand]

        hidden = torch.cat((protein_h, ligand_h), dim=0)
        positions = torch.cat((protein_pos.to(dtype=torch.float32), ligand_pos.to(dtype=torch.float32)), dim=0)
        batch = torch.cat((batch_protein, batch_ligand), dim=0)
        for edge_mlp, update_mlp in zip(self.edge_layers, self.update_layers):
            src, dst, distances = self._build_knn_edges(positions, batch)
            aggregate = torch.zeros_like(hidden)
            if src.numel():
                edge_input = torch.cat((hidden[dst], hidden[src], self._rbf(distances)), dim=-1)
                messages = edge_mlp(edge_input)
                aggregate.index_add_(0, dst, messages)
                counts = torch.zeros(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
                counts.index_add_(0, dst, torch.ones_like(distances))
                aggregate = aggregate / counts.clamp_min(1.0)[:, None]
            hidden = hidden + update_mlp(torch.cat((hidden, aggregate), dim=-1))
        return hidden[: protein_pos.shape[0]], hidden[protein_pos.shape[0] :]


__all__ = ["DistanceInvariantEncoder"]
