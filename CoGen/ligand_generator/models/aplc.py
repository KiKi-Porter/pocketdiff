from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from models.common import GaussianSmearing, MLP


def _empty_edges(device: torch.device) -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _same_type_edges(
    positions: torch.Tensor,
    global_indices: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if positions.size(0) <= 1:
        return _empty_edges(positions.device)
    k = min(k, positions.size(0) - 1)
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    neighbors = distances.topk(k=k, largest=False, dim=1).indices
    destinations = global_indices[:, None].expand(-1, k).reshape(-1)
    sources = global_indices[neighbors.reshape(-1)]
    return torch.stack([sources, destinations], dim=0)


def _cross_type_edges(
    source_positions: torch.Tensor,
    source_indices: torch.Tensor,
    destination_positions: torch.Tensor,
    destination_indices: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if source_positions.size(0) == 0 or destination_positions.size(0) == 0:
        return _empty_edges(destination_positions.device)
    k = min(k, source_positions.size(0))
    distances = torch.cdist(destination_positions, source_positions)
    neighbors = distances.topk(k=k, largest=False, dim=1).indices
    destinations = destination_indices[:, None].expand(-1, k).reshape(-1)
    sources = source_indices[neighbors.reshape(-1)]
    return torch.stack([sources, destinations], dim=0)


class _DirectionalMessage(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        radial_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.message = MLP(
            hidden_dim * 2 + radial_dim,
            output_dim,
            hidden_dim,
            norm=True,
            act_fn="silu",
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        radial: nn.Module,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return hidden.new_zeros((hidden.size(0), self.message.net[-1].out_features))
        source, destination = edge_index
        relative = positions[destination] - positions[source]
        distance = relative.norm(dim=-1, keepdim=True)
        radial_feature = radial(distance)
        message = self.message(
            torch.cat([hidden[source], hidden[destination], radial_feature], dim=-1)
        )
        return scatter_mean(
            message,
            destination,
            dim=0,
            dim_size=hidden.size(0),
        )


class APLCBlock(nn.Module):
    """Time-conditioned asymmetric pocket-ligand co-interaction residual.

    The block updates scalar hidden states only. Four separately parameterized
    streams preserve the direction of information flow while the original
    equivariant TargetDiff layer remains responsible for coordinate updates.
    A zero-initialized timestep modulation and layer gate let the block start
    from the previous single-site APLC behavior.
    """

    def __init__(
        self,
        hidden_dim: int,
        k: int = 16,
        num_r_gaussian: int = 16,
        r_max: float = 10.0,
        num_layers: int = 1,
        layerwise_layers: int = 1,
    ):
        super().__init__()
        self.k = k
        self.num_layers = max(1, int(num_layers))
        self.layerwise_layers = min(
            max(1, int(layerwise_layers)),
            self.num_layers,
        )
        self.radial = GaussianSmearing(
            0.0,
            r_max,
            num_gaussians=num_r_gaussian,
            fixed_offset=False,
        )
        self.protein_to_protein = _DirectionalMessage(
            hidden_dim, num_r_gaussian, hidden_dim
        )
        self.ligand_to_ligand = _DirectionalMessage(
            hidden_dim, num_r_gaussian, hidden_dim
        )
        self.protein_to_ligand = _DirectionalMessage(
            hidden_dim, num_r_gaussian, hidden_dim
        )
        self.ligand_to_protein = _DirectionalMessage(
            hidden_dim, num_r_gaussian, hidden_dim
        )
        self.protein_fusion = MLP(
            hidden_dim * 3,
            hidden_dim,
            hidden_dim,
            norm=True,
            act_fn="silu",
        )
        self.ligand_fusion = MLP(
            hidden_dim * 3,
            hidden_dim,
            hidden_dim,
            norm=True,
            act_fn="silu",
        )
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.layer_gates = nn.Parameter(torch.zeros(self.num_layers))
        with torch.no_grad():
            # Preserve the previous APLC behavior when loading an old
            # checkpoint: only the final layer is active initially.
            self.layer_gates[-1] = 1.0

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_scale = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.time_scale.weight)
        nn.init.zeros_(self.time_scale.bias)

    def _build_edges(
        self,
        positions: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pp_edges = []
        ll_edges = []
        pl_edges = []
        lp_edges = []
        for graph_id in range(int(batch.max().item()) + 1):
            graph_mask = batch == graph_id
            protein = torch.nonzero(graph_mask & ~mask_ligand, as_tuple=False).flatten()
            ligand = torch.nonzero(graph_mask & mask_ligand, as_tuple=False).flatten()
            pp_edges.append(
                _same_type_edges(positions[protein], protein, self.k)
            )
            ll_edges.append(
                _same_type_edges(positions[ligand], ligand, self.k)
            )
            pl_edges.append(
                _cross_type_edges(
                    positions[protein],
                    protein,
                    positions[ligand],
                    ligand,
                    self.k,
                )
            )
            lp_edges.append(
                _cross_type_edges(
                    positions[ligand],
                    ligand,
                    positions[protein],
                    protein,
                    self.k,
                )
            )

        def concatenate(edges: list[torch.Tensor]) -> torch.Tensor:
            nonempty = [edge for edge in edges if edge.numel()]
            return (
                torch.cat(nonempty, dim=1)
                if nonempty
                else _empty_edges(positions.device)
            )

        return (
            concatenate(pp_edges),
            concatenate(ll_edges),
            concatenate(pl_edges),
            concatenate(lp_edges),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
        time_step: torch.Tensor | None = None,
        num_timesteps: int | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        pp_edges, ll_edges, pl_edges, lp_edges = self._build_edges(
            positions,
            mask_ligand,
            batch,
        )
        pp_message = self.protein_to_protein(
            hidden, positions, pp_edges, self.radial
        )
        ll_message = self.ligand_to_ligand(
            hidden, positions, ll_edges, self.radial
        )
        pl_message = self.protein_to_ligand(
            hidden, positions, pl_edges, self.radial
        )
        lp_message = self.ligand_to_protein(
            hidden, positions, lp_edges, self.radial
        )

        protein_mask = ~mask_ligand
        ligand_mask = mask_ligand
        protein_delta = hidden.new_zeros(hidden.shape)
        ligand_delta = hidden.new_zeros(hidden.shape)
        protein_delta[protein_mask] = self.protein_fusion(
            torch.cat(
                [
                    hidden[protein_mask],
                    pp_message[protein_mask],
                    lp_message[protein_mask],
                ],
                dim=-1,
            )
        )
        ligand_delta[ligand_mask] = self.ligand_fusion(
            torch.cat(
                [
                    hidden[ligand_mask],
                    ll_message[ligand_mask],
                    pl_message[ligand_mask],
                ],
                dim=-1,
            )
        )
        delta = protein_delta + ligand_delta

        if time_step is not None:
            if num_timesteps is None or num_timesteps < 1:
                raise ValueError(
                    "num_timesteps is required for time-conditioned APLC"
                )
            if time_step.ndim != 1:
                raise ValueError("time_step must have shape [num_graphs]")
            if time_step.numel() == 0:
                raise ValueError("time_step must not be empty")
            normalized_time = (
                time_step.to(dtype=hidden.dtype).view(-1, 1)
                / float(max(1, num_timesteps - 1))
            )
            time_hidden = self.time_mlp(normalized_time)
            # The output projection is zero initialized, so old checkpoints
            # retain their original APLC behavior until this branch learns.
            time_scale = 0.25 * torch.tanh(self.time_scale(time_hidden))
            node_scale = hidden.new_zeros((hidden.size(0), 1))
            node_scale[protein_mask] = time_scale[batch[protein_mask], 0:1]
            node_scale[ligand_mask] = time_scale[batch[ligand_mask], 1:2]
            delta = delta * (1.0 + node_scale)

        if not 0 <= int(layer_index) < self.num_layers:
            raise ValueError(
                f"layer_index={layer_index} outside [0, {self.num_layers})"
            )
        layer_gate = self.layer_gates[int(layer_index)]
        return hidden + self.residual_gate.tanh() * layer_gate * delta
