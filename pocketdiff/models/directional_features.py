"""Residue-local directional features for the PocketDiff diffusion adapter.

The scalar distance encoder is useful for invariant context, but it cannot
tell a residue which direction the ligand lies in.  Apo2Mol and DynamicBind
solve this with vector/equivariant message passing and local-frame features.
This module provides the smallest compatible version of that signal: all
vectors are expressed in the current N--CA--C residue frame, so the features
are invariant to a global rigid transform while retaining directional
information relative to the ligand and neighbouring residues.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models.residue_pool import scatter_mean_residue


Tensor = torch.Tensor


def _validate_inputs(
    protein_pos: Tensor,
    atom_to_residue: Tensor,
    protein_atom_name: Sequence[str],
    frame_valid: Tensor,
    ligand_pos: Tensor,
    num_residues: int,
) -> None:
    if protein_pos.ndim != 2 or protein_pos.shape[-1] != 3:
        raise ValueError("protein_pos must have shape [N, 3]")
    if atom_to_residue.dtype != torch.long or atom_to_residue.shape != (protein_pos.shape[0],):
        raise ValueError("atom_to_residue must be LongTensor [N]")
    if len(protein_atom_name) != protein_pos.shape[0]:
        raise ValueError("protein_atom_name must describe every protein atom")
    if frame_valid.dtype != torch.bool or frame_valid.shape != (num_residues,):
        raise ValueError("frame_valid must be BoolTensor [Nr]")
    if ligand_pos.ndim != 2 or ligand_pos.shape[-1] != 3:
        raise ValueError("ligand_pos must have shape [Nl, 3]")
    if atom_to_residue.numel():
        if int(atom_to_residue.min()) < 0 or int(atom_to_residue.max()) >= num_residues:
            raise ValueError("atom_to_residue contains an out-of-range residue id")


def _to_local(global_vectors: Tensor, frames: Tensor) -> Tensor:
    """Convert row-vector global displacements into frame-local coordinates."""

    return torch.bmm(global_vectors.unsqueeze(1), frames).squeeze(1)


def build_residue_local_directional_features(
    protein_pos: Tensor,
    atom_to_residue: Tensor,
    protein_atom_name: Sequence[str],
    frame_valid: Tensor,
    ligand_pos: Tensor,
) -> Tensor:
    """Return 13 normalized residue-local directional features.

    The feature order is:

    1. ligand centroid displacement in the residue frame (3);
    2. bounded ligand centroid distance (1);
    3. nearest ligand atom displacement in the residue frame (3);
    4. residue atom-centroid displacement from the frame origin (3);
    5. nearest other-residue-origin displacement in the residue frame (3).

    Invalid frames and residues without a neighbour receive zeros.  All
    operations are differentiable with respect to current protein and ligand
    coordinates, while the topological nearest-neighbour selections are
    treated as discrete metadata for the current state.
    """

    num_residues = int(frame_valid.shape[0])
    _validate_inputs(
        protein_pos,
        atom_to_residue,
        protein_atom_name,
        frame_valid,
        ligand_pos,
        num_residues,
    )
    frames_result = build_residue_frames(
        protein_pos,
        atom_to_residue,
        protein_atom_name,
        num_residues=num_residues,
    )
    valid = frame_valid & frames_result.valid
    origins = frames_result.origins.to(dtype=torch.float32)
    frames = frames_result.frames.to(dtype=torch.float32)
    positions = protein_pos.to(dtype=torch.float32)
    ligand = ligand_pos.to(dtype=torch.float32)

    residue_centroids = scatter_mean_residue(
        positions,
        atom_to_residue,
        num_residues,
    )

    if ligand.shape[0] > 0:
        ligand_centroid = ligand.mean(dim=0)
        ligand_centroid_global = ligand_centroid.unsqueeze(0) - origins
        ligand_centroid_local = _to_local(ligand_centroid_global, frames)
        ligand_distance = torch.linalg.vector_norm(ligand_centroid_global, dim=-1)

        ligand_distances = torch.cdist(origins, ligand, p=2)
        nearest_ligand = ligand_distances.argmin(dim=1)
        nearest_ligand_global = ligand[nearest_ligand] - origins
        nearest_ligand_local = _to_local(nearest_ligand_global, frames)
    else:
        ligand_centroid_local = torch.zeros_like(origins)
        ligand_distance = torch.zeros(num_residues, dtype=origins.dtype, device=origins.device)
        nearest_ligand_local = torch.zeros_like(origins)

    residue_global = residue_centroids - origins
    residue_local = _to_local(residue_global, frames)

    if num_residues > 1:
        residue_distances = torch.cdist(origins, origins, p=2)
        residue_distances = residue_distances.masked_fill(
            torch.eye(num_residues, dtype=torch.bool, device=origins.device),
            float("inf"),
        )
        nearest_residue = residue_distances.argmin(dim=1)
        nearest_residue_global = origins[nearest_residue] - origins
        nearest_residue_local = _to_local(nearest_residue_global, frames)
    else:
        nearest_residue_local = torch.zeros_like(origins)

    # The scales are deliberately conservative for protein coordinates in Å.
    # tanh keeps outlier ligand distances from dominating the scalar heads.
    features = torch.cat(
        (
            torch.tanh(ligand_centroid_local / 10.0),
            torch.tanh(ligand_distance / 10.0).unsqueeze(-1),
            torch.tanh(nearest_ligand_local / 10.0),
            torch.tanh(residue_local / 4.0),
            torch.tanh(nearest_residue_local / 10.0),
        ),
        dim=-1,
    )
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(valid[:, None], features, torch.zeros_like(features))


def build_batched_residue_local_directional_features(
    protein_pos: Tensor,
    atom_to_residue: Tensor,
    protein_atom_name: Sequence[str],
    frame_valid: Tensor,
    ligand_pos: Tensor,
    batch_protein: Tensor,
    batch_residue: Tensor,
    batch_ligand: Tensor,
) -> Tensor:
    """Build directional features independently for each graph in a batch."""

    if batch_protein.dtype != torch.long or batch_protein.shape != (protein_pos.shape[0],):
        raise ValueError("batch_protein must be LongTensor [Np]")
    if batch_residue.dtype != torch.long or batch_residue.shape != (frame_valid.shape[0],):
        raise ValueError("batch_residue must be LongTensor [Nr]")
    if batch_ligand.dtype != torch.long or batch_ligand.shape != (ligand_pos.shape[0],):
        raise ValueError("batch_ligand must be LongTensor [Nl]")
    outputs = []
    residue_offset = 0
    graph_ids = torch.unique(
        torch.cat((batch_protein, batch_residue, batch_ligand)), sorted=True
    )
    for graph_id in graph_ids.tolist():
        protein_indices = torch.where(batch_protein == graph_id)[0]
        residue_indices = torch.where(batch_residue == graph_id)[0]
        ligand_indices = torch.where(batch_ligand == graph_id)[0]
        if residue_indices.numel() == 0:
            continue
        local_atom_to_residue = atom_to_residue[protein_indices] - int(residue_indices[0])
        local_features = build_residue_local_directional_features(
            protein_pos[protein_indices],
            local_atom_to_residue,
            [protein_atom_name[int(index)] for index in protein_indices.tolist()],
            frame_valid[residue_indices],
            ligand_pos[ligand_indices],
        )
        if int(residue_indices[0]) != residue_offset:
            raise ValueError(
                "batch_residue must be contiguous and ordered by graph for batched features"
            )
        outputs.append(local_features)
        residue_offset += int(residue_indices.numel())
    if residue_offset != frame_valid.shape[0]:
        raise ValueError("batch_residue does not cover every residue")
    return torch.cat(outputs, dim=0)


class LigandProteinVectorCrossMessage(nn.Module):
    """A small residue/ligand cross graph with a vector-valued message.

    The edge weight is a scalar learned from residue and ligand hidden states
    plus an RBF distance embedding.  The value carried by the edge is the
    ligand-to-residue displacement vector.  Aggregation happens in global
    coordinates and is converted to the current residue frame before being
    returned, so the result has the same global-rigid invariance as the
    DynamicBind local-frame features without requiring e3nn.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_rbf: int = 16,
        edge_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_rbf <= 0 or edge_hidden_dim <= 0:
            raise ValueError("hidden_dim, num_rbf and edge_hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_rbf = int(num_rbf)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
        )
        self.weight_head = nn.Linear(edge_hidden_dim, 1)
        self.contact_head = nn.Linear(edge_hidden_dim, 1)
        centers = torch.linspace(0.0, 20.0, num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_width = 20.0 / max(num_rbf - 1, 1)

    def _rbf(self, distances: Tensor) -> Tensor:
        centers = self.rbf_centers.to(dtype=distances.dtype, device=distances.device)
        return torch.exp(
            -((distances[..., None] - centers[None, :]) / self.rbf_width).square()
        )

    def forward(
        self,
        residue_hidden: Tensor,
        residue_origins: Tensor,
        residue_frames: Tensor,
        residue_valid: Tensor,
        ligand_hidden: Tensor,
        ligand_pos: Tensor,
    ) -> Tensor:
        if residue_hidden.ndim != 2 or residue_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"residue_hidden must have shape [Nr, {self.hidden_dim}]"
            )
        nr = residue_hidden.shape[0]
        if residue_origins.shape != (nr, 3):
            raise ValueError("residue_origins must have shape [Nr, 3]")
        if residue_frames.shape != (nr, 3, 3):
            raise ValueError("residue_frames must have shape [Nr, 3, 3]")
        if residue_valid.dtype != torch.bool or residue_valid.shape != (nr,):
            raise ValueError("residue_valid must be BoolTensor [Nr]")
        if ligand_hidden.ndim != 2 or ligand_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"ligand_hidden must have shape [Nl, {self.hidden_dim}]"
            )
        if ligand_pos.shape != (ligand_hidden.shape[0], 3):
            raise ValueError("ligand_pos and ligand_hidden must describe the same atoms")
        if ligand_hidden.shape[0] == 0:
            return residue_hidden.new_zeros((nr, 4))

        distances = torch.cdist(
            residue_origins.to(dtype=torch.float32),
            ligand_pos.to(dtype=torch.float32),
            p=2,
        )
        edge_input = torch.cat(
            (
                residue_hidden[:, None, :].expand(-1, ligand_hidden.shape[0], -1),
                ligand_hidden[None, :, :].expand(nr, -1, -1),
                self._rbf(distances.reshape(-1)).reshape(nr, ligand_hidden.shape[0], -1),
            ),
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_input.reshape(-1, edge_input.shape[-1]))
        scalar_weight = torch.sigmoid(self.weight_head(edge_hidden)).reshape(nr, -1)
        contact = torch.sigmoid(self.contact_head(edge_hidden)).reshape(nr, -1)
        displacement = ligand_pos[None, :, :] - residue_origins[:, None, :]
        unit_displacement = displacement / distances.clamp_min(1.0e-4)[..., None]
        normalization = scalar_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        global_vector = (scalar_weight[..., None] * unit_displacement).sum(dim=1) / normalization
        local_vector = torch.bmm(
            global_vector[:, None, :],
            residue_frames.to(dtype=global_vector.dtype),
        ).squeeze(1)
        contact_strength = contact.mean(dim=1, keepdim=True)
        output = torch.cat((local_vector, contact_strength), dim=-1)
        return torch.where(residue_valid[:, None], output, torch.zeros_like(output))

    def forward_batched(
        self,
        residue_hidden: Tensor,
        residue_origins: Tensor,
        residue_frames: Tensor,
        residue_valid: Tensor,
        ligand_hidden: Tensor,
        ligand_pos: Tensor,
        batch_residue: Tensor,
        batch_ligand: Tensor,
    ) -> Tensor:
        """Apply the cross message independently to each graph."""

        if batch_residue.dtype != torch.long or batch_residue.shape != (residue_hidden.shape[0],):
            raise ValueError("batch_residue must be LongTensor [Nr]")
        if batch_ligand.dtype != torch.long or batch_ligand.shape != (ligand_hidden.shape[0],):
            raise ValueError("batch_ligand must be LongTensor [Nl]")
        output = residue_hidden.new_zeros((residue_hidden.shape[0], 4))
        graph_ids = torch.unique(
            torch.cat((batch_residue, batch_ligand)), sorted=True
        )
        for graph_id in graph_ids.tolist():
            residue_indices = torch.where(batch_residue == graph_id)[0]
            ligand_indices = torch.where(batch_ligand == graph_id)[0]
            if residue_indices.numel() == 0:
                continue
            output[residue_indices] = self.forward(
                residue_hidden[residue_indices],
                residue_origins[residue_indices],
                residue_frames[residue_indices],
                residue_valid[residue_indices],
                ligand_hidden[ligand_indices],
                ligand_pos[ligand_indices],
            )
        return output


__all__ = [
    "LigandProteinVectorCrossMessage",
    "build_batched_residue_local_directional_features",
    "build_residue_local_directional_features",
]
