from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .geometry import residue_frames


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size,) + values.shape[1:])
    if index.numel():
        output.index_add_(0, index, values)
    return output


def _rbf(distance: torch.Tensor, count: int, cutoff: float) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, count, device=distance.device, dtype=distance.dtype)
    width = cutoff / max(count - 1, 1)
    return torch.exp(-((distance[:, None] - centers[None, :]) / width) ** 2)


def _bounded_vector(values: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values * (torch.tanh(norm) / norm.clamp_min(1e-8)) * maximum_norm


class ResidueVectorBlock(nn.Module):
    def __init__(self, hidden: int, vector_channels: int, radial: int):
        super().__init__()
        self.rr_message = nn.Sequential(
            nn.Linear(hidden * 2 + radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden + 2 * vector_channels),
        )
        self.lr_message = nn.Sequential(
            nn.Linear(hidden * 2 + radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden + 2 * vector_channels),
        )
        self.rl_message = nn.Sequential(
            nn.Linear(hidden * 2 + radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden + vector_channels),
        )
        self.ligand_node_update = nn.Sequential(
            nn.Linear(hidden * 2 + vector_channels, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.ligand_vector_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, vector_channels),
        )
        self.ligand_node_norm = nn.LayerNorm(hidden)
        self.node_update = nn.Sequential(
            nn.Linear(hidden * 2 + vector_channels, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.vector_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, vector_channels),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        ligand_scalar: torch.Tensor,
        ligand_vector: torch.Tensor,
        residue_pos: torch.Tensor,
        ligand_pos: torch.Tensor,
        rr_edge_index: torch.Tensor,
        lr_edge_index: torch.Tensor,
        radial_count: int,
        cutoff: float,
    ):
        nr = scalar.shape[0]
        scalar_agg = scalar.new_zeros(scalar.shape)
        vector_agg = vector.new_zeros(vector.shape)

        if rr_edge_index.shape[1]:
            source, target = rr_edge_index
            displacement = residue_pos[source] - residue_pos[target]
            distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1e-8)
            direction = displacement / distance[:, None]
            edge = torch.cat(
                (scalar[source], scalar[target], _rbf(distance, radial_count, cutoff)), dim=-1
            )
            messages = self.rr_message(edge)
            scalar_message = messages[:, : scalar.shape[-1]]
            direction_gate = messages[:, scalar.shape[-1] : scalar.shape[-1] + vector.shape[1]]
            source_gate = messages[:, scalar.shape[-1] + vector.shape[1] :]
            vector_message = (
                direction_gate[:, :, None] * direction[:, None, :]
                + source_gate[:, :, None] * vector[source]
            )
            scalar_agg.index_add_(0, target, scalar_message)
            vector_agg.index_add_(0, target, vector_message)

        if lr_edge_index.shape[1]:
            ligand_index, residue_index = lr_edge_index
            displacement = ligand_pos[ligand_index] - residue_pos[residue_index]
            distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1e-8)
            direction = displacement / distance[:, None]
            edge = torch.cat(
                (
                    scalar[residue_index],
                    ligand_scalar[ligand_index],
                    _rbf(distance, radial_count, cutoff),
                ),
                dim=-1,
            )
            messages = self.lr_message(edge)
            scalar_message = messages[:, : scalar.shape[-1]]
            vector_gate = messages[:, scalar.shape[-1] :]
            direction_gate = vector_gate[:, : vector.shape[1]]
            ligand_vector_gate = vector_gate[:, vector.shape[1] :]
            scalar_agg.index_add_(0, residue_index, scalar_message)
            vector_agg.index_add_(
                0,
                residue_index,
                direction_gate[:, :, None] * direction[:, None, :]
                + ligand_vector_gate[:, :, None] * ligand_vector[ligand_index],
            )

            reverse_edge = torch.cat(
                (
                    ligand_scalar[ligand_index],
                    scalar[residue_index],
                    _rbf(distance, radial_count, cutoff),
                ),
                dim=-1,
            )
            reverse_messages = self.rl_message(reverse_edge)
            ligand_scalar_message = reverse_messages[:, : scalar.shape[-1]]
            ligand_direction_gate = reverse_messages[:, scalar.shape[-1] :]
            ligand_scalar_aggregate = _scatter_sum(
                ligand_scalar_message, ligand_index, ligand_scalar.shape[0]
            )
            ligand_vector_aggregate = _scatter_sum(
                ligand_direction_gate[:, :, None] * (-direction[:, None, :])
                + vector[residue_index],
                ligand_index,
                ligand_scalar.shape[0],
            )
            ligand_vector_norm = torch.sqrt(
                (ligand_vector_aggregate * ligand_vector_aggregate).sum(-1) + 1e-8
            )
            ligand_scalar = self.ligand_node_norm(
                ligand_scalar
                + self.ligand_node_update(
                    torch.cat(
                        (
                            ligand_scalar,
                            ligand_scalar_aggregate,
                            ligand_vector_norm,
                        ),
                        dim=-1,
                    )
                )
            )
            ligand_vector = (
                ligand_vector
                + ligand_vector_aggregate
                + torch.tanh(self.ligand_vector_update(ligand_scalar))[:, :, None]
                * ligand_vector
            )

        vector_norm = torch.sqrt((vector_agg * vector_agg).sum(-1) + 1e-8)
        updated_scalar = self.norm(
            scalar + self.node_update(torch.cat((scalar, scalar_agg, vector_norm), dim=-1))
        )
        channel_gate = torch.tanh(self.vector_update(updated_scalar))
        updated_vector = vector + vector_agg + channel_gate[:, :, None] * vector
        return updated_scalar, updated_vector


class PocketDiffV4Model(nn.Module):
    """Residue-level scalar/vector field with SE(3)-equivariant coordinate heads."""

    def __init__(
        self,
        hidden: int = 192,
        vector_channels: int = 16,
        layers: int = 6,
        radial_count: int = 24,
        radial_cutoff: float = 16.0,
        max_translation: float = 1.0,
        max_rotation: float = 0.5,
        max_chi_step: float = 0.35,
    ):
        super().__init__()
        self.hidden = hidden
        self.vector_channels = vector_channels
        self.radial_count = radial_count
        self.radial_cutoff = radial_cutoff
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_chi_step = max_chi_step
        self.residue_embedding = nn.Embedding(20, 32)
        self.ligand_embedding = nn.Embedding(13, 48)
        self.residue_input = nn.Sequential(
            nn.Linear(27 + 32 + 3 + 1 + 1 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.ligand_input = nn.Sequential(
            nn.Linear(48 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.ligand_message = nn.Sequential(
            nn.Linear(hidden * 2 + radial_count, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden + vector_channels),
        )
        self.ligand_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.ligand_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList(
            ResidueVectorBlock(hidden, vector_channels, radial_count)
            for _ in range(layers)
        )
        self.translation_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, vector_channels))
        self.rotation_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, vector_channels))
        self.chi_head = nn.Sequential(
            nn.Linear(hidden + vector_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 5),
        )
        nn.init.zeros_(self.translation_gate[-1].weight)
        nn.init.zeros_(self.translation_gate[-1].bias)
        nn.init.zeros_(self.rotation_gate[-1].weight)
        nn.init.zeros_(self.rotation_gate[-1].bias)
        nn.init.zeros_(self.chi_head[-1].weight)
        nn.init.zeros_(self.chi_head[-1].bias)

    def forward(
        self,
        sample: Dict[str, object],
        protein_pos: torch.Tensor,
        time: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        inputs = sample.get("input", sample)
        device = protein_pos.device
        frame_index = inputs["frame_index"].to(device)
        atom_to_residue = inputs["atom_to_residue"].to(device)
        apo_pos = inputs["apo_pos"].to(device)
        residue_type = inputs["residue_type"].to(device)
        residue_feature = inputs["residue_feature"].to(device)
        ligand_pos = inputs["ligand_pos"].to(device)
        ligand_type = inputs["ligand_type"].to(device)
        rr_edge_index = inputs["rr_edge_index"].to(device)
        lr_edge_index = inputs["lr_edge_index"].to(device)
        ll_edge_index = inputs["ll_edge_index"].to(device)

        current_origin, frame, frame_valid = residue_frames(protein_pos, frame_index)
        apo_origin, _, apo_frame_valid = residue_frames(apo_pos, frame_index)
        local_displacement = torch.bmm(
            (current_origin - apo_origin).unsqueeze(1), frame
        ).squeeze(1)
        displacement_norm = torch.linalg.vector_norm(local_displacement, dim=-1, keepdim=True)
        time_value = torch.as_tensor(time, device=device, dtype=protein_pos.dtype).reshape(1, 1)
        time_feature = time_value.expand(current_origin.shape[0], 1)
        scalar_input = torch.cat(
            (
                residue_feature,
                self.residue_embedding(residue_type),
                local_displacement,
                displacement_norm,
                (frame_valid & apo_frame_valid).to(protein_pos.dtype)[:, None],
                time_feature,
            ),
            dim=-1,
        )
        scalar = self.residue_input(scalar_input)
        ligand_degree = torch.bincount(
            ll_edge_index[1], minlength=ligand_pos.shape[0]
        ).to(protein_pos.dtype)
        ligand_scalar = self.ligand_input(
            torch.cat(
                (
                    self.ligand_embedding(ligand_type),
                    torch.log1p(ligand_degree)[:, None],
                ),
                dim=-1,
            )
        )
        vector = protein_pos.new_zeros(
            (current_origin.shape[0], self.vector_channels, 3)
        )
        ligand_vector = protein_pos.new_zeros(
            (ligand_pos.shape[0], self.vector_channels, 3)
        )
        if ll_edge_index.shape[1]:
            ligand_source, ligand_target = ll_edge_index
            ligand_displacement = ligand_pos[ligand_source] - ligand_pos[ligand_target]
            ligand_distance = torch.linalg.vector_norm(
                ligand_displacement, dim=-1
            ).clamp_min(1e-8)
            ligand_direction = ligand_displacement / ligand_distance[:, None]
            ligand_edge = torch.cat(
                (
                    ligand_scalar[ligand_source],
                    ligand_scalar[ligand_target],
                    _rbf(ligand_distance, self.radial_count, 5.0),
                ),
                dim=-1,
            )
            ligand_messages = self.ligand_message(ligand_edge)
            ligand_scalar_message = ligand_messages[:, : self.hidden]
            ligand_vector_gate = ligand_messages[:, self.hidden :]
            ligand_scalar_aggregate = _scatter_sum(
                ligand_scalar_message, ligand_target, ligand_pos.shape[0]
            )
            ligand_vector.index_add_(
                0,
                ligand_target,
                ligand_vector_gate[:, :, None] * ligand_direction[:, None, :],
            )
            ligand_scalar = self.ligand_norm(
                ligand_scalar
                + self.ligand_update(
                    torch.cat((ligand_scalar, ligand_scalar_aggregate), dim=-1)
                )
            )
        for block in self.blocks:
            scalar, vector = block(
                scalar,
                vector,
                ligand_scalar,
                ligand_vector,
                current_origin,
                ligand_pos,
                rr_edge_index,
                lr_edge_index,
                self.radial_count,
                self.radial_cutoff,
            )

        translation_global = (
            torch.tanh(self.translation_gate(scalar))[:, :, None] * vector
        ).sum(dim=1)
        rotation_global = (
            torch.tanh(self.rotation_gate(scalar))[:, :, None] * vector
        ).sum(dim=1)
        translation_local = torch.bmm(
            translation_global.unsqueeze(1), frame
        ).squeeze(1)
        rotation_local = torch.bmm(
            rotation_global.unsqueeze(1), frame
        ).squeeze(1)
        chi_invariant = torch.sqrt((vector * vector).sum(-1) + 1e-8)
        chi_delta = torch.tanh(
            self.chi_head(torch.cat((scalar, chi_invariant), dim=-1))
        ) * self.max_chi_step
        valid = frame_valid & apo_frame_valid
        translation_local = torch.where(
            valid[:, None],
            _bounded_vector(translation_local, self.max_translation),
            torch.zeros_like(translation_local),
        )
        rotation_local = torch.where(
            valid[:, None],
            _bounded_vector(rotation_local, self.max_rotation),
            torch.zeros_like(rotation_local),
        )
        chi_geometry_mask = inputs["chi_geometry_mask"].to(device) & valid[:, None]
        chi_delta = torch.where(chi_geometry_mask, chi_delta, torch.zeros_like(chi_delta))
        return {
            "translation_local": translation_local,
            "rotation_local": rotation_local,
            "chi_delta": chi_delta,
            "frame_valid": valid,
        }
