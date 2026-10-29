from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum

from models.common import GaussianSmearing, MLP


def _empty_edges(device: torch.device) -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


class ConservativeCrossAdapter(nn.Module):
    """Bounded, equivariant protein-ligand output adapter.

    The original TargetDiff network remains the primary denoiser. This module
    only predicts a zero-initialized correction from protein-ligand cross
    pairs. Position corrections are weighted sums of relative unit vectors,
    so they transform equivariantly under global rotations and translations.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        k: int = 16,
        num_r_gaussian: int = 16,
        r_max: float = 10.0,
        max_pos_delta: float = 0.25,
        min_pos_delta: float = 0.02,
        type_scale: float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.k = int(k)
        self.max_pos_delta = float(max_pos_delta)
        self.min_pos_delta = float(min_pos_delta)
        self.type_scale = float(type_scale)

        self.radial = GaussianSmearing(
            0.0,
            r_max,
            num_gaussians=num_r_gaussian,
            fixed_offset=False,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        pair_dim = hidden_dim * 2 + num_r_gaussian + hidden_dim
        self.pair_context = MLP(
            pair_dim,
            hidden_dim,
            hidden_dim,
            norm=True,
            act_fn="silu",
        )
        self.direction_head = nn.Linear(pair_dim, 1)
        self.type_head = MLP(
            hidden_dim * 2,
            num_classes,
            hidden_dim,
            norm=True,
            act_fn="silu",
        )

        # The new path starts as an exact no-op on the original model.
        nn.init.zeros_(self.direction_head.weight)
        nn.init.zeros_(self.direction_head.bias)
        nn.init.zeros_(self.type_head.net[-1].weight)
        nn.init.zeros_(self.type_head.net[-1].bias)

    def _cross_edges(
        self,
        positions: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        edges = []
        num_graphs = int(batch.max().item()) + 1
        for graph_id in range(num_graphs):
            graph_mask = batch == graph_id
            protein = torch.nonzero(
                graph_mask & ~mask_ligand, as_tuple=False
            ).flatten()
            ligand = torch.nonzero(
                graph_mask & mask_ligand, as_tuple=False
            ).flatten()
            if protein.numel() == 0 or ligand.numel() == 0:
                continue
            k = min(self.k, int(protein.numel()))
            distances = torch.cdist(positions[ligand], positions[protein])
            neighbors = distances.topk(k=k, largest=False, dim=1).indices
            destinations = ligand[:, None].expand(-1, k).reshape(-1)
            sources = protein[neighbors.reshape(-1)]
            edges.append(torch.stack([sources, destinations], dim=0))
        if not edges:
            return _empty_edges(positions.device)
        return torch.cat(edges, dim=1)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
        time_step: torch.Tensor | None,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = self._cross_edges(positions, mask_ligand, batch)
        ligand_indices = torch.nonzero(mask_ligand, as_tuple=False).flatten()
        if edge_index.numel() == 0 or ligand_indices.numel() == 0:
            return (
                positions.new_zeros(positions.shape),
                hidden.new_zeros((hidden.size(0), self.num_classes)),
            )

        if time_step is None:
            normalized_time = hidden.new_zeros((int(batch.max().item()) + 1, 1))
        else:
            normalized_time = (
                time_step.to(dtype=hidden.dtype).view(-1, 1)
                / float(max(1, num_timesteps - 1))
            ).clamp(0.0, 1.0)
        time_hidden = self.time_mlp(normalized_time)

        source, destination = edge_index
        relative = positions[destination] - positions[source]
        distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        radial = self.radial(distance)
        pair_input = torch.cat(
            [
                hidden[source],
                hidden[destination],
                radial,
                time_hidden[batch[destination]],
            ],
            dim=-1,
        )

        pair_context = self.pair_context(pair_input)
        ligand_context = scatter_mean(
            pair_context,
            destination,
            dim=0,
            dim_size=hidden.size(0),
        )
        direction_weight = torch.tanh(self.direction_head(pair_input)).squeeze(-1)
        unit_relative = relative / distance
        delta_pos = scatter_sum(
            direction_weight.unsqueeze(-1) * unit_relative,
            destination,
            dim=0,
            dim_size=hidden.size(0),
        )

        t_node = time_hidden[batch]
        type_input = torch.cat([hidden, ligand_context + t_node], dim=-1)
        delta_type = self.type_head(type_input)

        t_ligand = normalized_time[batch]
        position_scale = (
            self.min_pos_delta
            + (self.max_pos_delta - self.min_pos_delta) * t_ligand
        )
        delta_pos = delta_pos * position_scale
        delta_pos = delta_pos * mask_ligand.unsqueeze(-1)
        delta_type = (
            self.type_scale
            * torch.tanh(delta_type)
            * mask_ligand.unsqueeze(-1)
        )
        return delta_pos, delta_type
