"""Small DynamicBind-style scalar/vector residue message passing block.

The official DynamicBind model uses e3nn tensor products.  The development
environment for PocketDiff does not provide e3nn, so this module keeps the
same useful architectural split in plain PyTorch:

* scalar residue/ligand states are updated on residue-residue and
  ligand-residue graphs;
* vector states are built from relative displacement directions and are
  updated with scalar gates;
* the final vector field is converted to the current residue frame before it
  reaches the motion head.

All vector operations are equivariant under a common rotation/translation:
relative vectors rotate with the input and scalar gates are invariant.  This
is deliberately a compact development backend, not a replacement for the
official DynamicBind implementation.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


Tensor = torch.Tensor


def _knn_edges(positions: Tensor, *, knn: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Return directed residue edges ``src, dst, distance`` for one graph."""

    count = int(positions.shape[0])
    if count < 2:
        empty_long = torch.empty(0, dtype=torch.long, device=positions.device)
        empty_float = positions.new_empty((0,))
        return empty_long, empty_long, empty_float
    distances = torch.cdist(positions, positions, p=2)
    distances = distances.masked_fill(
        torch.eye(count, dtype=torch.bool, device=positions.device),
        float("inf"),
    )
    neighbors = min(int(knn), count - 1)
    nearest_distances, nearest_src = torch.topk(
        distances, k=neighbors, dim=1, largest=False, sorted=False
    )
    dst = torch.arange(count, device=positions.device, dtype=torch.long)
    dst = dst[:, None].expand(count, neighbors).reshape(-1)
    src = nearest_src.reshape(-1)
    return src, dst, nearest_distances.reshape(-1)


class DynamicBindVectorBlock(nn.Module):
    """A compact residue scalar/vector update field.

    Parameters are intentionally modest because the current trainer processes
    one complex per update.  ``residue_hidden`` and ``ligand_hidden`` retain
    the 128-dimensional contract used by the existing PocketDiff heads.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        time_dim: int = 128,
        vector_channels: int = 4,
        num_layers: int = 2,
        knn: int = 24,
        num_rbf: int = 16,
        edge_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(
            hidden_dim,
            time_dim,
            vector_channels,
            num_layers,
            knn,
            num_rbf,
            edge_hidden_dim,
        ) <= 0:
            raise ValueError("DynamicBindVectorBlock dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.vector_channels = int(vector_channels)
        self.num_layers = int(num_layers)
        self.knn = int(knn)
        self.num_rbf = int(num_rbf)
        self.edge_hidden_dim = int(edge_hidden_dim)

        centers = torch.linspace(0.0, 20.0, num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_width = 20.0 / max(num_rbf - 1, 1)

        edge_input_dim = 2 * hidden_dim + time_dim + num_rbf
        cross_input_dim = 2 * hidden_dim + time_dim + num_rbf
        self.residue_edge_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        )
        self.cross_edge_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(cross_input_dim, edge_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        )
        self.residue_scalar_message = nn.ModuleList(
            nn.Linear(edge_hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.cross_scalar_message = nn.ModuleList(
            nn.Linear(edge_hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.residue_vector_message = nn.ModuleList(
            nn.Linear(edge_hidden_dim, vector_channels) for _ in range(num_layers)
        )
        self.cross_vector_message = nn.ModuleList(
            nn.Linear(edge_hidden_dim, vector_channels) for _ in range(num_layers)
        )
        self.residue_contact = nn.ModuleList(
            nn.Linear(edge_hidden_dim, 1) for _ in range(num_layers)
        )
        self.cross_contact = nn.ModuleList(
            nn.Linear(edge_hidden_dim, 1) for _ in range(num_layers)
        )
        self.vector_mix = nn.ModuleList(
            nn.Linear(vector_channels, vector_channels, bias=False)
            for _ in range(num_layers)
        )
        update_input_dim = 2 * hidden_dim + time_dim + vector_channels
        self.scalar_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(update_input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.vector_gates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(update_input_dim, vector_channels),
                nn.Sigmoid(),
            )
            for _ in range(num_layers)
        )
        self.vector_readout = nn.Linear(hidden_dim, vector_channels)

    def _rbf(self, distances: Tensor) -> Tensor:
        if distances.numel() == 0:
            return distances.new_empty((0, self.num_rbf))
        centers = self.rbf_centers.to(dtype=distances.dtype, device=distances.device)
        return torch.exp(
            -((distances[:, None] - centers[None, :]) / self.rbf_width).square()
        )

    @staticmethod
    def _aggregate(values: Tensor, dst: Tensor, count: int) -> Tensor:
        result = values.new_zeros((count,) + tuple(values.shape[1:]))
        if dst.numel():
            result.index_add_(0, dst, values)
            counts = values.new_zeros((count,))
            counts.index_add_(0, dst, torch.ones_like(dst, dtype=values.dtype))
            result = result / counts.clamp_min(1.0).reshape(
                (count,) + (1,) * (values.ndim - 1)
            )
        return result

    def forward(
        self,
        residue_hidden: Tensor,
        residue_origins: Tensor,
        ligand_hidden: Tensor,
        ligand_pos: Tensor,
        time_hidden: Tensor,
        residue_valid: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if residue_hidden.ndim != 2 or residue_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"residue_hidden must have shape [Nr, {self.hidden_dim}]"
            )
        nr = residue_hidden.shape[0]
        if residue_origins.shape != (nr, 3):
            raise ValueError("residue_origins must have shape [Nr, 3]")
        if ligand_hidden.ndim != 2 or ligand_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"ligand_hidden must have shape [Nl, {self.hidden_dim}]"
            )
        if ligand_pos.shape != (ligand_hidden.shape[0], 3):
            raise ValueError("ligand_pos and ligand_hidden must have matching shapes")
        if time_hidden.ndim == 1:
            time_hidden = time_hidden.unsqueeze(0)
        if time_hidden.shape != (1, self.time_dim):
            raise ValueError(f"time_hidden must have shape [1, {self.time_dim}]")
        if residue_valid.dtype != torch.bool or residue_valid.shape != (nr,):
            raise ValueError("residue_valid must be BoolTensor [Nr]")

        h = residue_hidden
        vector = residue_hidden.new_zeros((nr, self.vector_channels, 3))
        time_residue = time_hidden.expand(nr, -1)
        contact = residue_hidden.new_zeros((nr, 1))
        residue_src, residue_dst, residue_dist = _knn_edges(
            residue_origins.to(dtype=torch.float32), knn=self.knn
        )

        for layer in range(self.num_layers):
            scalar_residue = residue_hidden.new_zeros((nr, self.hidden_dim))
            vector_residue = residue_hidden.new_zeros(
                (nr, self.vector_channels, 3)
            )
            scalar_cross = residue_hidden.new_zeros((nr, self.hidden_dim))
            vector_cross = residue_hidden.new_zeros((nr, self.vector_channels, 3))
            layer_contact = residue_hidden.new_zeros((nr, 1))

            if residue_src.numel():
                edge_input = torch.cat(
                    (
                        h[residue_dst],
                        h[residue_src],
                        time_residue[residue_dst],
                        self._rbf(residue_dist),
                    ),
                    dim=-1,
                )
                edge_hidden = self.residue_edge_mlps[layer](edge_input)
                scalar_residue = self._aggregate(
                    self.residue_scalar_message[layer](edge_hidden),
                    residue_dst,
                    nr,
                )
                direction = (
                    residue_origins[residue_src] - residue_origins[residue_dst]
                )
                direction = direction / residue_dist.clamp_min(1.0e-4)[:, None]
                coefficients = torch.tanh(
                    self.residue_vector_message[layer](edge_hidden)
                )
                direction_message = coefficients[:, :, None] * direction[:, None, :]
                source_vector = self.vector_mix[layer](
                    vector[residue_src].transpose(1, 2)
                ).transpose(1, 2)
                vector_residue = self._aggregate(
                    direction_message + 0.25 * source_vector,
                    residue_dst,
                    nr,
                )
                layer_contact = self._aggregate(
                    torch.sigmoid(self.residue_contact[layer](edge_hidden)),
                    residue_dst,
                    nr,
                )

            if ligand_hidden.shape[0]:
                cross_dist = torch.cdist(
                    residue_origins.to(dtype=torch.float32),
                    ligand_pos.to(dtype=torch.float32),
                    p=2,
                )
                cross_dst = torch.arange(
                    nr, device=residue_hidden.device, dtype=torch.long
                )[:, None].expand(nr, ligand_hidden.shape[0]).reshape(-1)
                cross_src = torch.arange(
                    ligand_hidden.shape[0],
                    device=residue_hidden.device,
                    dtype=torch.long,
                )[None, :].expand(nr, -1).reshape(-1)
                cross_dist_flat = cross_dist.reshape(-1)
                cross_input = torch.cat(
                    (
                        h[cross_dst],
                        ligand_hidden[cross_src],
                        time_residue[cross_dst],
                        self._rbf(cross_dist_flat),
                    ),
                    dim=-1,
                )
                cross_hidden = self.cross_edge_mlps[layer](cross_input)
                scalar_cross = self._aggregate(
                    self.cross_scalar_message[layer](cross_hidden),
                    cross_dst,
                    nr,
                )
                direction = (
                    ligand_pos[cross_src] - residue_origins[cross_dst]
                )
                direction = direction / cross_dist_flat.clamp_min(1.0e-4)[:, None]
                coefficients = torch.tanh(
                    self.cross_vector_message[layer](cross_hidden)
                )
                vector_cross = self._aggregate(
                    coefficients[:, :, None] * direction[:, None, :],
                    cross_dst,
                    nr,
                )
                layer_contact = layer_contact + self._aggregate(
                    torch.sigmoid(self.cross_contact[layer](cross_hidden)),
                    cross_dst,
                    nr,
                )

            scalar_total = scalar_residue + scalar_cross
            vector_total = vector_residue + vector_cross
            vector_norms = torch.linalg.vector_norm(vector_total, dim=-1)
            update_input = torch.cat(
                (h, scalar_total, time_residue, vector_norms),
                dim=-1,
            )
            h = h + self.scalar_updates[layer](update_input)
            gate = self.vector_gates[layer](update_input)
            vector = vector + gate[:, :, None] * vector_total
            contact = 0.5 * contact + 0.5 * layer_contact

        h = torch.where(residue_valid[:, None], h, torch.zeros_like(h))
        vector = torch.where(residue_valid[:, None, None], vector, torch.zeros_like(vector))
        contact = torch.where(residue_valid[:, None], contact, torch.zeros_like(contact))
        channel_weights = torch.softmax(self.vector_readout(h), dim=-1)
        global_vector = (channel_weights[:, :, None] * vector).sum(dim=1)
        return h, global_vector, contact.clamp(0.0, 1.0)

    def forward_batched(
        self,
        residue_hidden: Tensor,
        residue_origins: Tensor,
        ligand_hidden: Tensor,
        ligand_pos: Tensor,
        time_hidden: Tensor,
        residue_valid: Tensor,
        batch_residue: Tensor,
        batch_ligand: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply DynamicBind messages independently to each graph."""

        if batch_residue.dtype != torch.long or batch_residue.shape != (residue_hidden.shape[0],):
            raise ValueError("batch_residue must be LongTensor [Nr]")
        if batch_ligand.dtype != torch.long or batch_ligand.shape != (ligand_hidden.shape[0],):
            raise ValueError("batch_ligand must be LongTensor [Nl]")
        if time_hidden.ndim != 2:
            raise ValueError("time_hidden must have shape [B, time_dim]")
        hidden_out = residue_hidden.new_zeros(residue_hidden.shape)
        vector_out = residue_hidden.new_zeros((residue_hidden.shape[0], 3))
        contact_out = residue_hidden.new_zeros((residue_hidden.shape[0], 1))
        graph_ids = torch.unique(
            torch.cat((batch_residue, batch_ligand)), sorted=True
        )
        for graph_id in graph_ids.tolist():
            residue_indices = torch.where(batch_residue == graph_id)[0]
            ligand_indices = torch.where(batch_ligand == graph_id)[0]
            if residue_indices.numel() == 0:
                continue
            graph_time = time_hidden[int(graph_id)].unsqueeze(0)
            hidden, vector, contact = self.forward(
                residue_hidden[residue_indices],
                residue_origins[residue_indices],
                ligand_hidden[ligand_indices],
                ligand_pos[ligand_indices],
                graph_time,
                residue_valid[residue_indices],
            )
            hidden_out[residue_indices] = hidden
            vector_out[residue_indices] = vector
            contact_out[residue_indices] = contact
        return hidden_out, vector_out, contact_out


__all__ = ["DynamicBindVectorBlock"]
