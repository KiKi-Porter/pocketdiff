from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum

from models.common import GaussianSmearing, MLP


class LayerwiseCrossInteraction(nn.Module):
    """A zero-initialized, time-conditioned protein-ligand residual branch.

    The original O(2) attention layer remains unchanged. This branch consumes
    only cross-type edges from that layer, updates scalar states for both node
    types, and updates ligand coordinates from relative vectors. Its outputs
    are gated separately so a checkpoint from the unmodified TargetDiff model
    remains an exact functional no-op before fine-tuning.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int = 4,
        num_r_gaussian: int = 16,
        r_max: float = 10.0,
        max_coord_delta: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.max_coord_delta = float(max_coord_delta)
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
        pair_dim = hidden_dim * 2 + edge_feat_dim + num_r_gaussian + hidden_dim
        self.hidden_message = MLP(
            pair_dim,
            hidden_dim,
            hidden_dim,
            norm=True,
            act_fn="silu",
        )
        self.coord_message = MLP(
            pair_dim,
            1,
            hidden_dim,
            norm=True,
            act_fn="silu",
        )
        self.hidden_gate = nn.Parameter(torch.zeros(()))
        self.coord_gate = nn.Parameter(torch.zeros(()))

    def _time_features(
        self,
        batch: torch.Tensor,
        time_step: torch.Tensor | None,
        num_timesteps: int | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1
        if time_step is None:
            normalized = torch.zeros(
                (num_graphs, 1),
                dtype=dtype,
                device=batch.device,
            )
        else:
            if num_timesteps is None or num_timesteps < 1:
                raise ValueError(
                    "num_timesteps is required for layerwise cross interaction"
                )
            normalized = (
                time_step.to(dtype=dtype).view(-1, 1)
                / float(max(1, num_timesteps - 1))
            ).clamp(0.0, 1.0)
        return self.time_mlp(normalized)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        time_step: torch.Tensor | None,
        num_timesteps: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        cross = mask_ligand[source] != mask_ligand[destination]
        if not bool(cross.any()):
            return torch.zeros_like(h), torch.zeros_like(x)

        source = source[cross]
        destination = destination[cross]
        cross_edge_attr = edge_attr[cross]
        relative = x[destination] - x[source]
        distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        radial = self.radial(distance)
        time_hidden = self._time_features(
            batch=batch,
            time_step=time_step,
            num_timesteps=num_timesteps,
            dtype=h.dtype,
        )
        pair_input = torch.cat(
            [
                h[source],
                h[destination],
                cross_edge_attr.to(dtype=h.dtype),
                radial,
                time_hidden[batch[destination]],
            ],
            dim=-1,
        )
        hidden_message = self.hidden_message(pair_input)
        delta_h = scatter_mean(
            hidden_message,
            destination,
            dim=0,
            dim_size=h.size(0),
        )

        # Only ligand destinations receive coordinate updates, matching
        # TargetDiff's original protein-fixed sampling protocol.
        ligand_destination = mask_ligand[destination]
        coord_weight = torch.tanh(
            self.coord_message(pair_input).squeeze(-1)
        )
        coord_weight = coord_weight * ligand_destination.to(coord_weight.dtype)
        delta_x = scatter_sum(
            coord_weight.unsqueeze(-1) * relative / distance,
            destination,
            dim=0,
            dim_size=x.size(0),
        )
        delta_x = (
            self.max_coord_delta
            * torch.tanh(self.coord_gate)
            * delta_x
        )
        delta_h = torch.tanh(self.hidden_gate) * delta_h
        return delta_h, delta_x
