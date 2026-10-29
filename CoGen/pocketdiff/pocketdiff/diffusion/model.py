"""Minimal continuous-time residue diffusion adapter for Phase 45."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from pocketdiff.diffusion.state import DiffusionStateInput, DiffusionStateTarget
from pocketdiff.models.invariant_encoder import DistanceInvariantEncoder
from pocketdiff.models.targetdiff_encoder import TargetDiffEncoderAdapter
from pocketdiff.models.motion_head import ResidueMotionHead
from pocketdiff.models.chi_head import ResidueChiHead
from pocketdiff.models.residue_pool import scatter_mean_residue
from pocketdiff.diffusion.se3 import apply_local_se3_update
from pocketdiff.geometry.chi import apply_chi_updates, build_chi_update_metadata
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models.directional_features import (
    LigandProteinVectorCrossMessage,
    build_batched_residue_local_directional_features,
    build_residue_local_directional_features,
)
from pocketdiff.models.dynamicbind_vector import DynamicBindVectorBlock
from pocketdiff.geometry.so3 import so3_log


@dataclass(frozen=True)
class DiffusionMotionPrediction:
    translation_local: torch.Tensor
    rotation_local: torch.Tensor
    chi: torch.Tensor


@dataclass(frozen=True)
class DiffusionMotionLoss:
    loss: torch.Tensor
    translation_loss: torch.Tensor
    rotation_loss: torch.Tensor
    chi_loss: torch.Tensor
    endpoint_loss: torch.Tensor
    backbone_loss: torch.Tensor
    next_state_loss: torch.Tensor
    continuity_loss: torch.Tensor
    direction_loss: torch.Tensor
    backbone_progress_direction_loss: torch.Tensor
    backbone_progress_magnitude_loss: torch.Tensor


class _PeptideSequenceMessage(nn.Module):
    """Chain-local scalar message for adjacent peptide residues."""

    def __init__(self, hidden_dim: int, output_dim: int = 16) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        batch_residue: torch.Tensor,
        residue_position: torch.Tensor,
        chain_break: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(hidden)
        has_neighbor = torch.zeros(
            hidden.shape[0], dtype=torch.bool, device=hidden.device
        )
        indices = torch.arange(hidden.shape[0], device=hidden.device)
        left = indices - 1
        right = indices + 1
        left_safe = left.clamp_min(0)
        right_safe = right.clamp_max(hidden.shape[0] - 1)
        left_valid = (
            (indices > 0)
            & (batch_residue[left_safe] == batch_residue)
            & (chain_break < 0.5)
        )
        right_valid = (
            (right < hidden.shape[0])
            & (batch_residue[right_safe] == batch_residue)
            & (chain_break[right_safe] < 0.5)
        )
        if bool(left_valid.any()):
            aggregate[left_valid] += hidden[left_safe[left_valid]]
            has_neighbor[left_valid] = True
        if bool(right_valid.any()):
            aggregate[right_valid] += hidden[right_safe[right_valid]]
            has_neighbor[right_valid] = True
        degree = left_valid.to(hidden.dtype) + right_valid.to(hidden.dtype)
        aggregate = aggregate / degree.clamp_min(1.0)[:, None]
        features = torch.stack(
            (
                torch.sin(residue_position / 10.0),
                torch.cos(residue_position / 10.0),
                torch.sin(residue_position / 50.0),
                chain_break,
            ),
            dim=-1,
        )
        updated = self.update(torch.cat((hidden, aggregate, features), dim=-1))
        updated = torch.where(has_neighbor[:, None], hidden + updated, hidden)
        return self.readout(updated)


class _ResidueAtomAttention(nn.Module):
    """Cheap topology-preserving atom-to-residue attention pooling.

    TargetDiff has already mixed atom-level protein/ligand contacts into each
    protein atom hidden state. A learned positive weight keeps the most
    contact-relevant atoms from being washed out by an unweighted residue
    mean, while retaining one fixed residue node per downstream SE(3) head.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        atom_hidden: torch.Tensor,
        atom_to_residue: torch.Tensor,
        num_residues: int,
    ) -> torch.Tensor:
        if atom_hidden.ndim != 2 or atom_to_residue.ndim != 1:
            raise ValueError("atom hidden and atom_to_residue must be rank 2/1")
        logits = self.score(atom_hidden).squeeze(-1).clamp(-12.0, 12.0)
        weights = torch.nn.functional.softplus(logits) + 1.0e-4
        pooled = torch.zeros(
            (num_residues, atom_hidden.shape[1]),
            dtype=atom_hidden.dtype,
            device=atom_hidden.device,
        )
        pooled.index_add_(0, atom_to_residue, atom_hidden * weights[:, None])
        normalizer = torch.zeros(
            num_residues, dtype=atom_hidden.dtype, device=atom_hidden.device
        )
        normalizer.index_add_(0, atom_to_residue, weights)
        return pooled / normalizer.clamp_min(1.0e-6)[:, None]


class _AtomLevelLigandContactMessage(nn.Module):
    """Protein-atom/ligand-atom contact field for residue motion.

    TargetDiff mixes the joint atom graph into hidden states, but the v4.6
    motion adapter then reduced the ligand condition mostly to residue origins.
    This branch keeps the final contact readout atom-level: every protein atom
    can receive a ligand-conditioned scalar message and a local directional
    vector before the result is pooled back to residues.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        output_dim: int = 16,
        edge_hidden_dim: int = 96,
        num_rbf: int = 16,
        cutoff: float = 10.0,
    ) -> None:
        super().__init__()
        if min(hidden_dim, output_dim, edge_hidden_dim, num_rbf) <= 0:
            raise ValueError("contact message dimensions must be positive")
        if cutoff <= 0.0:
            raise ValueError("contact cutoff must be positive")
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf + 1, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
        )
        self.scalar_message = nn.Linear(edge_hidden_dim, output_dim)
        self.vector_weight = nn.Linear(edge_hidden_dim, 1)
        self.contact_weight = nn.Linear(edge_hidden_dim, 1)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_width = cutoff / max(num_rbf - 1, 1)

    def _rbf(self, distance: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.to(
            device=distance.device,
            dtype=distance.dtype,
        )
        return torch.exp(
            -((distance[..., None] - centers[None, :]) / self.rbf_width).square()
        )

    def forward_batched(
        self,
        protein_hidden: torch.Tensor,
        protein_pos: torch.Tensor,
        ligand_hidden: torch.Tensor,
        ligand_pos: torch.Tensor,
        atom_to_residue: torch.Tensor,
        residue_origins: torch.Tensor,
        residue_frames: torch.Tensor,
        residue_valid: torch.Tensor,
        batch_protein: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_residue: torch.Tensor,
    ) -> torch.Tensor:
        if protein_hidden.shape != (protein_pos.shape[0], self.hidden_dim):
            raise ValueError("protein_hidden has an invalid shape")
        if ligand_hidden.shape != (ligand_pos.shape[0], self.hidden_dim):
            raise ValueError("ligand_hidden has an invalid shape")
        if atom_to_residue.shape != (protein_pos.shape[0],):
            raise ValueError("atom_to_residue must cover every protein atom")
        output = protein_hidden.new_zeros(
            (residue_origins.shape[0], self.output_dim + 4)
        )
        graph_ids = torch.unique(
            torch.cat((batch_protein, batch_ligand, batch_residue)),
            sorted=True,
        )
        for graph_id in graph_ids.tolist():
            protein_indices = torch.where(batch_protein == graph_id)[0]
            ligand_indices = torch.where(batch_ligand == graph_id)[0]
            residue_indices = torch.where(batch_residue == graph_id)[0]
            if not protein_indices.numel() or not ligand_indices.numel():
                continue
            residue_start = int(residue_indices[0])
            local_atom_to_residue = (
                atom_to_residue[protein_indices] - residue_start
            )
            distances = torch.cdist(
                protein_pos[protein_indices].to(dtype=torch.float32),
                ligand_pos[ligand_indices].to(dtype=torch.float32),
                p=2,
            )
            edge_mask = distances <= self.cutoff
            if not bool(edge_mask.any()):
                continue
            protein_local = protein_hidden[protein_indices]
            ligand_local = ligand_hidden[ligand_indices]
            pair_protein, pair_ligand = torch.where(edge_mask)
            pair_distance = distances[pair_protein, pair_ligand]
            edge_input = torch.cat(
                (
                    protein_local[pair_protein],
                    ligand_local[pair_ligand],
                    self._rbf(pair_distance),
                    torch.tanh(pair_distance[:, None] / self.cutoff),
                ),
                dim=-1,
            )
            edge_hidden = self.edge_mlp(edge_input)
            weight = torch.sigmoid(self.contact_weight(edge_hidden)).squeeze(-1)
            scalar = self.scalar_message(edge_hidden) * weight[:, None]
            atom_scalar = protein_hidden.new_zeros(
                (protein_indices.shape[0], self.output_dim)
            )
            atom_scalar.index_add_(0, pair_protein, scalar)
            atom_norm = protein_hidden.new_zeros((protein_indices.shape[0],))
            atom_norm.index_add_(0, pair_protein, weight)
            atom_scalar = atom_scalar / atom_norm.clamp_min(1.0e-6)[:, None]

            displacement = (
                ligand_pos[ligand_indices[pair_ligand]]
                - protein_pos[protein_indices[pair_protein]]
            ).to(dtype=protein_hidden.dtype)
            unit_displacement = displacement / pair_distance.to(
                dtype=displacement.dtype
            ).clamp_min(1.0e-4)[:, None]
            vector_coeff = torch.tanh(
                self.vector_weight(edge_hidden).squeeze(-1)
            ) * weight
            atom_vector = protein_hidden.new_zeros(
                (protein_indices.shape[0], 3)
            )
            atom_vector.index_add_(
                0,
                pair_protein,
                vector_coeff[:, None] * unit_displacement,
            )
            atom_vector = atom_vector / atom_norm.clamp_min(1.0e-6)[:, None]
            atom_contact = protein_hidden.new_zeros((protein_indices.shape[0],))
            atom_contact.index_add_(0, pair_protein, weight)
            atom_contact = atom_contact / atom_norm.clamp_min(1.0e-6)

            residue_scalar = protein_hidden.new_zeros(
                (residue_indices.shape[0], self.output_dim)
            )
            residue_scalar.index_add_(0, local_atom_to_residue, atom_scalar)
            residue_count = protein_hidden.new_zeros((residue_indices.shape[0],))
            residue_count.index_add_(
                0,
                local_atom_to_residue,
                torch.ones_like(atom_norm),
            )
            residue_scalar = residue_scalar / residue_count.clamp_min(1.0)[:, None]
            residue_vector = protein_hidden.new_zeros(
                (residue_indices.shape[0], 3)
            )
            residue_vector.index_add_(0, local_atom_to_residue, atom_vector)
            residue_vector = residue_vector / residue_count.clamp_min(1.0)[:, None]
            local_vector = torch.bmm(
                residue_vector[:, None, :],
                residue_frames[residue_indices].to(dtype=residue_vector.dtype),
            ).squeeze(1)
            residue_contact = protein_hidden.new_zeros(
                (residue_indices.shape[0],)
            )
            residue_contact.index_add_(0, local_atom_to_residue, atom_contact)
            residue_contact = residue_contact / residue_count.clamp_min(1.0)
            graph_output = torch.cat(
                (residue_scalar, local_vector, residue_contact[:, None]),
                dim=-1,
            )
            graph_output = torch.where(
                residue_valid[residue_indices, None],
                graph_output,
                torch.zeros_like(graph_output),
            )
            output[residue_indices] = graph_output
        return output


class _SequenceMotionCoupling(nn.Module):
    """Learned sequence-edge context used only by the motion head."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        contact_dim: int = 20,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 * contact_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        residue_hidden: torch.Tensor,
        contact_hidden: torch.Tensor,
        batch_residue: torch.Tensor,
        residue_position: torch.Tensor,
        chain_break: torch.Tensor,
    ) -> torch.Tensor:
        count = residue_hidden.shape[0]
        aggregate_h = torch.zeros_like(residue_hidden)
        aggregate_contact = torch.zeros_like(contact_hidden)
        has_neighbor = torch.zeros(count, dtype=torch.bool, device=residue_hidden.device)
        indices = torch.arange(count, device=residue_hidden.device)
        left = indices - 1
        right = indices + 1
        left_safe = left.clamp_min(0)
        right_safe = right.clamp_max(max(count - 1, 0))
        left_valid = (
            (indices > 0)
            & (batch_residue[left_safe] == batch_residue)
            & (chain_break < 0.5)
        )
        right_valid = (
            (right < count)
            & (batch_residue[right_safe] == batch_residue)
            & (chain_break[right_safe] < 0.5)
        )
        for valid, neighbor in ((left_valid, left_safe), (right_valid, right_safe)):
            if bool(valid.any()):
                aggregate_h[valid] += residue_hidden[neighbor[valid]]
                aggregate_contact[valid] += contact_hidden[neighbor[valid]]
                has_neighbor[valid] = True
        degree = left_valid.to(residue_hidden.dtype) + right_valid.to(residue_hidden.dtype)
        aggregate_h = aggregate_h / degree.clamp_min(1.0)[:, None]
        aggregate_contact = aggregate_contact / degree.clamp_min(1.0)[:, None]
        edge_features = torch.stack(
            (
                torch.sin(residue_position / 10.0),
                torch.cos(residue_position / 10.0),
                torch.sin(residue_position / 50.0),
                torch.cos(residue_position / 50.0),
                chain_break,
            ),
            dim=-1,
        )
        updated = self.update(
            torch.cat(
                (
                    residue_hidden,
                    aggregate_h,
                    contact_hidden,
                    aggregate_contact,
                    edge_features,
                ),
                dim=-1,
            )
        )
        updated = torch.where(has_neighbor[:, None], updated, torch.zeros_like(updated))
        return self.readout(updated)


class _SequenceMotionConstraint(nn.Module):
    """Chain-local gated smoothing for residue SE(3) velocity."""

    def __init__(self, condition_dim: int = 16) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(condition_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(
        self,
        translation_local: torch.Tensor,
        rotation_local: torch.Tensor,
        frames: torch.Tensor,
        batch_residue: torch.Tensor,
        chain_break: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = translation_local.shape[0]
        if count == 0:
            return translation_local, rotation_local
        indices = torch.arange(count, device=translation_local.device)
        left = indices - 1
        right = indices + 1
        left_safe = left.clamp_min(0)
        right_safe = right.clamp_max(count - 1)
        left_valid = (
            (indices > 0)
            & (batch_residue[left_safe] == batch_residue)
            & (chain_break < 0.5)
        )
        right_valid = (
            (right < count)
            & (batch_residue[right_safe] == batch_residue)
            & (chain_break[right_safe] < 0.5)
        )
        degree = left_valid.to(translation_local.dtype) + right_valid.to(
            translation_local.dtype
        )
        if not bool((degree > 0).any()):
            return translation_local, rotation_local

        global_translation = torch.bmm(
            translation_local[:, None, :],
            frames.transpose(-1, -2),
        ).squeeze(1)
        global_rotation = torch.bmm(
            rotation_local[:, None, :],
            frames.transpose(-1, -2),
        ).squeeze(1)
        neighbor_translation = torch.zeros_like(global_translation)
        neighbor_rotation = torch.zeros_like(global_rotation)
        if bool(left_valid.any()):
            neighbor_translation[left_valid] += global_translation[left_safe[left_valid]]
            neighbor_rotation[left_valid] += global_rotation[left_safe[left_valid]]
        if bool(right_valid.any()):
            neighbor_translation[right_valid] += global_translation[right_safe[right_valid]]
            neighbor_rotation[right_valid] += global_rotation[right_safe[right_valid]]
        neighbor_translation = neighbor_translation / degree.clamp_min(1.0)[:, None]
        neighbor_rotation = neighbor_rotation / degree.clamp_min(1.0)[:, None]
        gate = self.gate(condition) * (degree > 0).to(condition.dtype)[:, None]
        global_translation = global_translation + gate * (
            neighbor_translation - global_translation
        )
        global_rotation = global_rotation + gate * (
            neighbor_rotation - global_rotation
        )
        constrained_translation = torch.bmm(
            global_translation[:, None, :],
            frames,
        ).squeeze(1)
        constrained_rotation = torch.bmm(
            global_rotation[:, None, :],
            frames,
        ).squeeze(1)
        return constrained_translation, constrained_rotation


def _residue_graph_time(
    model_input: DiffusionStateInput,
    residue_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    times = model_input.diffusion_time.reshape(-1).to(device=device, dtype=dtype)
    batch_residue = model_input.batch_residue
    if batch_residue is None:
        batch_residue = torch.zeros(
            residue_count, dtype=torch.long, device=device
        )
    batch_residue = batch_residue.to(device=device)
    if batch_residue.shape != (residue_count,):
        raise ValueError("batch_residue must match residue count")
    if times.numel() == 0 or int(batch_residue.max()) >= times.numel():
        raise ValueError("diffusion_time does not cover all residues")
    return times[batch_residue]


def _backbone_atom_mask(model_input: DiffusionStateInput) -> torch.Tensor:
    names = tuple(model_input.protein_atom_name)
    return torch.tensor(
        [name in {"N", "CA", "C", "O"} for name in names],
        dtype=torch.bool,
        device=model_input.protein_pos.device,
    )


def _backbone_progress_loss(
    predicted: torch.Tensor,
    oracle_next: torch.Tensor,
    model_input: DiffusionStateInput,
    target: DiffusionStateTarget,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise directed CA progress toward the holo bridge.

    The mask is defined from the apo-to-holo CA displacement, while the
    target step is the label-independent oracle bridge next state.  This
    prevents an unconditioned ``move away from apo`` objective from rewarding
    arbitrary motion and keeps the loss valid for backbone-only training.
    """

    names = tuple(model_input.protein_atom_name)
    ca_mask = torch.tensor(
        [name == "CA" for name in names],
        dtype=torch.bool,
        device=predicted.device,
    )
    apo_ref = model_input.apo_pos_ref
    if apo_ref is None or not bool(ca_mask.any()):
        zero = predicted.sum() * 0.0
        return zero, zero
    apo_ref = apo_ref.to(device=predicted.device, dtype=predicted.dtype)
    holo = target.protein_pos_holo.to(device=predicted.device, dtype=predicted.dtype)
    current_ca = model_input.protein_pos[ca_mask]
    predicted_ca = predicted[ca_mask]
    oracle_ca = oracle_next[ca_mask].detach()
    apo_ca = apo_ref[ca_mask]
    holo_ca = holo[ca_mask]
    atom_residue = model_input.atom_to_residue.to(device=predicted.device)
    residue_valid = target.target_valid & model_input.frame_valid
    valid_ca = residue_valid[atom_residue[ca_mask]]
    total_motion = torch.linalg.vector_norm(holo_ca - apo_ca, dim=-1)
    active = valid_ca & (total_motion > float(threshold))
    if not bool(active.any()):
        zero = predicted.sum() * 0.0
        return zero, zero

    predicted_delta = predicted_ca[active] - current_ca[active]
    oracle_delta = oracle_ca[active] - current_ca[active]
    oracle_norm = torch.linalg.vector_norm(oracle_delta, dim=-1)
    valid_step = oracle_norm > 1.0e-8
    if not bool(valid_step.any()):
        zero = predicted.sum() * 0.0
        return zero, zero
    predicted_delta = predicted_delta[valid_step]
    oracle_delta = oracle_delta[valid_step]
    oracle_norm = oracle_norm[valid_step]
    predicted_norm = torch.linalg.vector_norm(predicted_delta, dim=-1)
    cosine = (predicted_delta * oracle_delta).sum(dim=-1) / (
        predicted_norm.clamp_min(1.0e-6) * oracle_norm.clamp_min(1.0e-6)
    )
    direction_loss = (1.0 - cosine).mean()
    # Normalize by a conservative CA step scale so that the magnitude term is
    # comparable across t values without allowing a zero prediction to evade
    # supervision.  The target remains the oracle bridge step, not final holo.
    magnitude_loss = (
        (predicted_norm - oracle_norm) / oracle_norm.detach().clamp_min(0.05)
    ).square().mean()
    return direction_loss, magnitude_loss


def _chain_continuity_loss(
    predicted: torch.Tensor,
    holo: torch.Tensor,
    model_input: DiffusionStateInput,
) -> torch.Tensor:
    """Penalize chain-level backbone geometry drift.

    Residue SE(3) updates preserve all intra-residue distances, but independent
    residue motions can still stretch peptide links or distort local backbone
    angles.  The target structure supplies the physically valid local geometry;
    this loss only constrains those local invariants and does not prescribe the
    global apo-to-holo displacement.
    """

    def angle(first: torch.Tensor, vertex: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        first_vector = first - vertex
        last_vector = last - vertex
        denominator = (
            torch.linalg.vector_norm(first_vector).clamp_min(1.0e-6)
            * torch.linalg.vector_norm(last_vector).clamp_min(1.0e-6)
        )
        cosine = (first_vector * last_vector).sum() / denominator
        return torch.acos(cosine.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))

    atom_to_residue = model_input.atom_to_residue.detach().cpu().tolist()
    names = tuple(model_input.protein_atom_name)
    by_residue = {}
    for atom_id, residue_id in enumerate(atom_to_residue):
        by_residue.setdefault(residue_id, {})[names[atom_id]] = atom_id
    residue_count = len(by_residue)
    distances_pred = []
    distances_holo = []
    chain_break = model_input.residue_chain_break
    chain_break_values = (
        chain_break.detach().cpu().tolist()
        if chain_break is not None
        else [1.0] + [0.0] * max(residue_count - 1, 0)
    )
    for residue_id in range(residue_count - 1):
        if chain_break_values[residue_id + 1] > 0.5:
            continue
        left = by_residue.get(residue_id, {})
        right = by_residue.get(residue_id + 1, {})
        if "CA" in left and "CA" in right:
            distances_pred.append(
                torch.linalg.vector_norm(
                    predicted[left["CA"]] - predicted[right["CA"]]
                )
            )
            distances_holo.append(
                torch.linalg.vector_norm(holo[left["CA"]] - holo[right["CA"]])
            )
        if "C" in left and "N" in right:
            distances_pred.append(
                torch.linalg.vector_norm(
                    predicted[left["C"]] - predicted[right["N"]]
                )
            )
            distances_holo.append(
                torch.linalg.vector_norm(holo[left["C"]] - holo[right["N"]])
            )
    angle_residuals = []
    for residue_id in range(residue_count):
        current = by_residue.get(residue_id, {})
        if all(name in current for name in ("N", "CA", "C")):
            predicted_angle = angle(
                predicted[current["N"]],
                predicted[current["CA"]],
                predicted[current["C"]],
            )
            holo_angle = angle(
                holo[current["N"]],
                holo[current["CA"]],
                holo[current["C"]],
            )
            angle_residuals.append(predicted_angle - holo_angle)
        if residue_id >= residue_count - 1:
            continue
        if chain_break_values[residue_id + 1] > 0.5:
            continue
        left = by_residue.get(residue_id, {})
        right = by_residue.get(residue_id + 1, {})
        if all(name in left for name in ("CA", "C")) and "N" in right:
            predicted_angle = angle(
                predicted[left["CA"]],
                predicted[left["C"]],
                predicted[right["N"]],
            )
            holo_angle = angle(
                holo[left["CA"]],
                holo[left["C"]],
                holo[right["N"]],
            )
            angle_residuals.append(predicted_angle - holo_angle)
        if "C" in left and all(name in right for name in ("N", "CA")):
            predicted_angle = angle(
                predicted[left["C"]],
                predicted[right["N"]],
                predicted[right["CA"]],
            )
            holo_angle = angle(
                holo[left["C"]],
                holo[right["N"]],
                holo[right["CA"]],
            )
            angle_residuals.append(predicted_angle - holo_angle)
    if not distances_pred and not angle_residuals:
        return predicted.sum() * 0.0
    residuals = [
        (a - b).square() for a, b in zip(distances_pred, distances_holo)
    ]
    residuals.extend(value.square() for value in angle_residuals)
    return torch.stack(residuals).mean()


class DiffusionMotionAdapter(nn.Module):
    """Residue diffusion adapter with scalar, TargetDiff, and vector backends."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        encoder_backend: str = "scalar",
        prediction_type: str = "velocity",
        architecture_variant: str = "v45",
    ):
        super().__init__()
        if encoder_backend not in ("scalar", "targetdiff", "dynamicbind"):
            raise ValueError(
                "encoder_backend must be scalar, targetdiff, or dynamicbind"
            )
        if encoder_backend in ("targetdiff", "dynamicbind") and hidden_dim != 128:
            raise ValueError(
                "targetdiff and dynamicbind backends require hidden_dim=128"
            )
        self.encoder_backend = encoder_backend
        if prediction_type not in ("remaining", "velocity", "bridge_velocity"):
            raise ValueError("prediction_type must be remaining, velocity, or bridge_velocity")
        if architecture_variant not in ("v45", "v46", "v47"):
            raise ValueError("architecture_variant must be v45, v46, or v47")
        self.prediction_type = prediction_type
        self.architecture_variant = architecture_variant
        if encoder_backend in ("scalar", "dynamicbind"):
            self.encoder = DistanceInvariantEncoder(hidden_dim=hidden_dim)
            cross_num_rbf = self.encoder.num_rbf
        else:
            self.encoder = TargetDiffEncoderAdapter()
            cross_num_rbf = self.encoder.config["num_rbf"]
        if architecture_variant == "v47":
            self.descriptor_dim = 341
        elif architecture_variant == "v46":
            self.descriptor_dim = 305
        else:
            self.descriptor_dim = 289
        self.motion_head = ResidueMotionHead(
            descriptor_dim=self.descriptor_dim,
            dropout=dropout,
            zero_init=(
                encoder_backend != "dynamicbind"
                and architecture_variant != "v47"
            ),
            max_translation=2.5 if architecture_variant == "v47" else None,
        )
        self.chi_head = ResidueChiHead(
            descriptor_dim=self.descriptor_dim,
            dropout=dropout,
            zero_init=encoder_backend != "dynamicbind",
        )
        self.sequence_condition = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 16),
            nn.SiLU(),
            nn.Linear(16, 6),
        )
        self.sequence_message = (
            _PeptideSequenceMessage(hidden_dim)
            if architecture_variant in ("v46", "v47")
            else None
        )
        self.atom_contact_message = (
            _AtomLevelLigandContactMessage(hidden_dim=hidden_dim)
            if architecture_variant == "v47"
            else None
        )
        self.sequence_motion_coupling = (
            _SequenceMotionCoupling(hidden_dim=hidden_dim, contact_dim=20)
            if architecture_variant == "v47"
            else None
        )
        self.sequence_motion_constraint = (
            _SequenceMotionConstraint(condition_dim=16)
            if architecture_variant == "v47"
            else None
        )
        self.atom_pool = _ResidueAtomAttention(hidden_dim)
        # The legacy motion/χ heads intentionally keep their 283-dimensional
        # contract.  Six learned channels are reserved for a compact
        # projection of explicit residue-local directional geometry and a
        # ligand→protein vector cross message.
        self.directional_condition = nn.Sequential(
            nn.LayerNorm(23),
            nn.Linear(23, 32),
            nn.SiLU(),
            nn.Linear(32, 6),
        )
        self.vector_cross_message = LigandProteinVectorCrossMessage(
            hidden_dim=hidden_dim,
            num_rbf=cross_num_rbf,
        )
        self.dynamicbind_vector = (
            DynamicBindVectorBlock(hidden_dim=hidden_dim)
            if encoder_backend == "dynamicbind"
            else None
        )
        self.time_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        if encoder_backend != "dynamicbind":
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, value: DiffusionStateInput) -> DiffusionMotionPrediction:
        pos = value.protein_pos
        batch_protein = (
            value.batch_protein
            if value.batch_protein is not None
            else torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        )
        batch_ligand = (
            value.batch_ligand
            if value.batch_ligand is not None
            else torch.zeros(value.ligand_pos.shape[0], dtype=torch.long, device=pos.device)
        )
        batch_residue = (
            value.batch_residue
            if value.batch_residue is not None
            else torch.zeros(value.residue_type.shape[0], dtype=torch.long, device=pos.device)
        )
        if batch_protein.shape != (pos.shape[0],) or batch_ligand.shape != (value.ligand_pos.shape[0],):
            raise ValueError("batch metadata must match protein and ligand atom counts")
        if batch_residue.shape != (value.residue_type.shape[0],):
            raise ValueError("batch_residue must match residue count")
        if any(item.dtype != torch.long for item in (batch_protein, batch_residue, batch_ligand)):
            raise TypeError("batch metadata must be LongTensor")
        if batch_protein.numel() and int(batch_protein.min()) < 0:
            raise ValueError("batch_protein cannot contain negative graph ids")
        if batch_residue.numel() and int(batch_residue.min()) < 0:
            raise ValueError("batch_residue cannot contain negative graph ids")
        if batch_ligand.numel() and int(batch_ligand.min()) < 0:
            raise ValueError("batch_ligand cannot contain negative graph ids")
        graph_ids = torch.unique(
            torch.cat((batch_protein, batch_residue, batch_ligand)), sorted=True
        )
        if graph_ids.numel() == 0:
            raise ValueError("diffusion model input contains no graph nodes")
        if not torch.equal(
            graph_ids,
            torch.arange(graph_ids.numel(), device=graph_ids.device, dtype=torch.long),
        ):
            raise ValueError("batch graph ids must be contiguous starting at zero")
        if value.atom_to_residue.shape != (pos.shape[0],):
            raise ValueError("atom_to_residue must map every protein atom")
        if value.atom_to_residue.numel() and (
            int(value.atom_to_residue.min()) < 0
            or int(value.atom_to_residue.max()) >= batch_residue.numel()
        ):
            raise ValueError("atom_to_residue contains an out-of-range residue id")
        residue_graph_for_atom = batch_residue[value.atom_to_residue]
        if not torch.equal(batch_protein, residue_graph_for_atom):
            raise ValueError("atom_to_residue crosses graph boundaries")
        t = value.diffusion_time.reshape(-1).to(dtype=torch.float32)
        if t.numel() != graph_ids.numel():
            raise ValueError("diffusion_time must contain one value per graph")
        if not torch.isfinite(t).all():
            raise ValueError("diffusion_time must be finite")
        half = 32
        freq = torch.exp(torch.arange(half, dtype=torch.float32, device=pos.device) * (-torch.log(torch.tensor(10000.0, device=pos.device)) / (half - 1)))
        emb = torch.cat((torch.sin(t[:, None] * freq[None, :]), torch.cos(t[:, None] * freq[None, :])), dim=-1)
        time_h = self.time_mlp(emb)
        if self.encoder_backend in ("scalar", "dynamicbind"):
            protein_h, ligand_h = self.encoder(
                pos,
                value.protein_feature,
                batch_protein,
                value.ligand_pos,
                value.ligand_type,
                batch_ligand,
                time_h,
                time_h,
            )
        else:
            encoded = self.encoder(
                pos,
                value.protein_feature,
                batch_protein,
                value.ligand_pos,
                value.ligand_type,
                batch_ligand,
            )
            protein_h = encoded["protein_hidden"]
            ligand_h = encoded["ligand_hidden"]
            # The official backbone receives geometry and atom identity. Add
            # PocketDiff's continuous diffusion-time condition after the
            # frozen TargetDiff input contract, preserving the same 128-d
            # hidden width and current motion/χ heads.
            protein_h = protein_h + time_h[batch_protein]
            ligand_h = ligand_h + time_h[batch_ligand]
        residue_h = self.atom_pool(
            protein_h,
            value.atom_to_residue,
            value.residue_type.shape[0],
        )
        residue_one_hot = torch.nn.functional.one_hot(value.residue_type, num_classes=20).float()
        current_frames = build_residue_frames(
            pos,
            value.atom_to_residue,
            value.protein_atom_name,
            num_residues=value.residue_type.shape[0],
            atom_indices=value.frame_atom_indices,
        )
        frame_valid = value.frame_valid & current_frames.valid
        directional = build_batched_residue_local_directional_features(
            pos,
            value.atom_to_residue,
            value.protein_atom_name,
            frame_valid,
            value.ligand_pos,
            batch_protein,
            batch_residue,
            batch_ligand,
        )
        apo_pos = value.apo_pos_ref if value.apo_pos_ref is not None else pos
        if apo_pos.shape != pos.shape:
            raise ValueError("apo_pos_ref must match protein_pos")
        apo_frames = build_residue_frames(
            apo_pos,
            value.atom_to_residue,
            value.protein_atom_name,
            num_residues=value.residue_type.shape[0],
            atom_indices=value.frame_atom_indices,
        )
        anchor_valid = frame_valid & current_frames.valid & apo_frames.valid
        anchor_translation = torch.bmm(
            (apo_frames.origins - current_frames.origins).unsqueeze(1),
            current_frames.frames,
        ).squeeze(1)
        anchor_rotation = so3_log(
            current_frames.frames.transpose(-1, -2) @ apo_frames.frames
        )
        anchor = torch.cat((anchor_translation, anchor_rotation), dim=-1)
        anchor = torch.where(anchor_valid[:, None], anchor, torch.zeros_like(anchor))
        if self.encoder_backend == "dynamicbind":
            assert self.dynamicbind_vector is not None
            residue_h, global_vector, contact = self.dynamicbind_vector.forward_batched(
                residue_h,
                current_frames.origins,
                ligand_h,
                value.ligand_pos,
                time_h,
                frame_valid,
                batch_residue,
                batch_ligand,
            )
            local_vector = torch.bmm(
                global_vector[:, None, :],
                current_frames.frames.to(dtype=global_vector.dtype),
            ).squeeze(1)
            cross_directional = torch.cat(
                (
                    torch.tanh(local_vector / 4.0),
                    contact,
                ),
                dim=-1,
            )
        else:
            cross_directional = self.vector_cross_message.forward_batched(
                residue_h,
                current_frames.origins,
                current_frames.frames,
                frame_valid,
                ligand_h,
                value.ligand_pos,
                batch_residue,
                batch_ligand,
            )
        contact_condition = None
        if self.atom_contact_message is not None:
            contact_condition = self.atom_contact_message.forward_batched(
                protein_h,
                pos,
                ligand_h,
                value.ligand_pos,
                value.atom_to_residue,
                current_frames.origins,
                current_frames.frames,
                frame_valid,
                batch_protein,
                batch_ligand,
                batch_residue,
            )
        directional_h = self.directional_condition(
            torch.cat((directional, cross_directional, anchor), dim=-1)
        )
        if value.residue_position is None:
            residue_position = torch.zeros(
                (residue_h.shape[0],), dtype=pos.dtype, device=pos.device
            )
        else:
            residue_position = value.residue_position.to(
                device=pos.device, dtype=pos.dtype
            )
        if value.residue_chain_break is None:
            chain_break = torch.zeros_like(residue_position)
        else:
            chain_break = value.residue_chain_break.to(
                device=pos.device, dtype=pos.dtype
            )
        if residue_position.shape != (residue_h.shape[0],):
            raise ValueError("residue_position must match residue count")
        if chain_break.shape != residue_position.shape:
            raise ValueError("residue_chain_break must match residue count")
        sequence_features = torch.stack(
            (
                torch.sin(residue_position / 10.0),
                torch.cos(residue_position / 10.0),
                torch.sin(residue_position / 50.0),
                chain_break,
            ),
            dim=-1,
        )
        sequence_h = self.sequence_condition(sequence_features)
        descriptor_parts = [
            residue_h,
            residue_one_hot,
            value.frame_valid[:, None].float(),
            time_h[batch_residue],
            directional_h,
            sequence_h,
        ]
        if self.sequence_message is not None:
            descriptor_parts.append(
                self.sequence_message(
                    residue_h,
                    batch_residue,
                    residue_position,
                    chain_break,
                )
            )
        motion_condition = None
        if self.sequence_motion_coupling is not None:
            if contact_condition is None:
                raise RuntimeError("v47 sequence coupling requires contact condition")
            motion_condition = self.sequence_motion_coupling(
                residue_h,
                contact_condition,
                batch_residue,
                residue_position,
                chain_break,
            )
            descriptor_parts.extend(
                (
                    contact_condition,
                    motion_condition,
                )
            )
        descriptor = torch.cat(tuple(descriptor_parts), dim=-1)
        if descriptor.shape[-1] != self.descriptor_dim:
            raise RuntimeError(
                "DiffusionMotionAdapter descriptor contract changed: "
                f"expected {self.descriptor_dim}, got {descriptor.shape[-1]}"
            )
        translation, rotation = self.motion_head(descriptor)
        if self.sequence_motion_constraint is not None:
            if motion_condition is None:
                raise RuntimeError("v47 motion constraint requires sequence condition")
            translation, rotation = self.sequence_motion_constraint(
                translation,
                rotation,
                current_frames.frames,
                batch_residue,
                chain_break,
                motion_condition,
            )
        chi = self.chi_head(descriptor, value.chi_current, value.chi_mask)
        return DiffusionMotionPrediction(translation, rotation, chi)


def predict_holo_coordinates(
    model_input: DiffusionStateInput,
    prediction: DiffusionMotionPrediction,
    *,
    apply_chi: bool = True,
) -> torch.Tensor:
    """Apply one full predicted motion to the current noisy coordinates.

    The diffusion target is a remaining transform from the current state to
    holo.  This helper therefore applies the prediction with fraction one,
    using the same row-vector local-frame convention and rigid-then-χ order as
    the reverse sampler.  It is intended for a training-only endpoint loss;
    holo coordinates never enter ``model_input``.
    """
    protein_pos = model_input.protein_pos
    num_residues = int(model_input.residue_type.shape[0])
    if protein_pos.ndim != 2 or protein_pos.shape[-1] != 3:
        raise ValueError("model_input.protein_pos must have shape [N, 3]")
    if model_input.atom_to_residue.shape != (protein_pos.shape[0],):
        raise ValueError("model_input.atom_to_residue must map every protein atom")
    expected_translation = (num_residues, 3)
    if prediction.translation_local.shape != expected_translation:
        raise ValueError(
            "prediction.translation_local must have shape "
            f"{expected_translation}, got {tuple(prediction.translation_local.shape)}"
        )
    if prediction.rotation_local.shape != expected_translation:
        raise ValueError(
            "prediction.rotation_local must have shape "
            f"{expected_translation}, got {tuple(prediction.rotation_local.shape)}"
        )
    if prediction.chi.shape != (num_residues, 5):
        raise ValueError(
            "prediction.chi must have shape "
            f"{(num_residues, 5)}, got {tuple(prediction.chi.shape)}"
        )
    if model_input.frame_valid.shape != (num_residues,):
        raise ValueError("model_input.frame_valid must have shape [Nr]")
    if model_input.chi_mask.shape != (num_residues, 5):
        raise ValueError("model_input.chi_mask must have shape [Nr, 5]")

    current_frames = build_residue_frames(
        protein_pos,
        model_input.atom_to_residue,
        model_input.protein_atom_name,
        num_residues=num_residues,
        atom_indices=model_input.frame_atom_indices,
    )
    frame_valid = model_input.frame_valid & current_frames.valid
    rigid = apply_local_se3_update(
        protein_pos,
        model_input.atom_to_residue,
        current_frames.origins,
        current_frames.frames,
        prediction.translation_local,
        prediction.rotation_local,
        frame_valid=frame_valid,
    )

    if not apply_chi:
        return rigid
    # Rebuild the χ axes after the rigid update.  This matches the sampler:
    # each step applies residue-frame motion first and side-chain rotations
    # second, with χ topology derived from the current coordinates.
    chi_state = build_current_chi_state(
        rigid,
        model_input.protein_atom_name,
        model_input.atom_to_residue,
        model_input.protein_residue_name,
    )
    chi_valid = (
        model_input.chi_mask
        & chi_state.geometry_rotatable_mask
        & frame_valid[:, None]
    )
    return apply_chi_updates(
        rigid,
        chi_state.axis_start,
        chi_state.axis_end,
        chi_state.downstream_atom_mask,
        prediction.chi,
        valid=chi_valid,
    ).positions


def diffusion_endpoint_loss(
    prediction: DiffusionMotionPrediction,
    target: DiffusionStateTarget,
    model_input: DiffusionStateInput,
    *,
    backbone_only: bool = False,
) -> torch.Tensor:
    """Measure the predicted one-update coordinate endpoint against holo.

    The residue supervision mask is expanded to atoms through
    ``atom_to_residue``.  Invalid residues contribute no coordinate error, so
    missing/degenerate frame records cannot create a false endpoint signal.
    """
    predicted_holo = predict_holo_coordinates(
        model_input, prediction, apply_chi=not backbone_only
    )
    valid_residue = target.target_valid & model_input.frame_valid
    atom_valid = valid_residue[model_input.atom_to_residue]
    if backbone_only:
        atom_valid = atom_valid & _backbone_atom_mask(model_input)
    if not bool(atom_valid.any()):
        return predicted_holo.sum() * 0.0
    return (
        predicted_holo[atom_valid] - target.protein_pos_holo[atom_valid]
    ).square().mean()


def _scale_prediction(
    prediction: DiffusionMotionPrediction,
    fraction: torch.Tensor,
    *,
    backbone_only: bool,
) -> DiffusionMotionPrediction:
    scaled = DiffusionMotionPrediction(
        translation_local=prediction.translation_local * fraction[:, None],
        rotation_local=prediction.rotation_local * fraction[:, None],
        chi=prediction.chi * fraction[:, None],
    )
    if backbone_only:
        scaled = DiffusionMotionPrediction(
            translation_local=scaled.translation_local,
            rotation_local=scaled.rotation_local,
            chi=torch.zeros_like(scaled.chi),
        )
    return scaled


def _target_dt_per_residue(
    target: DiffusionStateTarget,
    model_input: DiffusionStateInput,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return bridge ``dt`` expanded from graph-level batch metadata."""

    if target.target_dt is None:
        return None
    batch_residue = model_input.batch_residue
    if batch_residue is None:
        batch_residue = torch.zeros(
            target.target_valid.shape[0], dtype=torch.long, device=device
        )
    else:
        batch_residue = batch_residue.to(device=device, dtype=torch.long)
    raw = target.target_dt
    if isinstance(raw, torch.Tensor):
        raw = raw.to(device=device, dtype=dtype).reshape(-1)
        if raw.numel() == 1:
            return raw.expand(target.target_valid.shape[0])
        if batch_residue.numel() and int(batch_residue.max()) >= raw.numel():
            raise ValueError("target_dt tensor does not cover all residue graphs")
        return raw[batch_residue]
    return torch.full(
        (target.target_valid.shape[0],),
        float(raw),
        dtype=dtype,
        device=device,
    )


def diffusion_next_state_loss(
    prediction: DiffusionMotionPrediction,
    target: DiffusionStateTarget,
    model_input: DiffusionStateInput,
    *,
    prediction_type: str,
    fixed_steps: int = 20,
    backbone_only: bool = True,
) -> torch.Tensor:
    """Compare one learned Euler step with the exact bridge next state.

    The model is trained on a continuous state at time ``t`` while deployment
    uses fixed Euler intervals.  For a remaining transform, the oracle bridge
    advances by ``dt / t``; for a velocity head the learned update advances by
    ``dt``.  This keeps the local training target identical to the 20-step
    sampler instead of supervising only a final, unreachable endpoint.
    """

    if fixed_steps <= 0:
        raise ValueError("fixed_steps must be positive")
    if prediction_type not in ("remaining", "velocity", "bridge_velocity"):
        raise ValueError("unsupported prediction_type")
    batch_residue = model_input.batch_residue
    if batch_residue is None:
        batch_residue = torch.zeros(
            target.target_valid.shape[0],
            dtype=torch.long,
            device=prediction.translation_local.device,
        )
    graph_time = model_input.diffusion_time.reshape(-1).to(
        device=prediction.translation_local.device,
        dtype=prediction.translation_local.dtype,
    )[batch_residue]
    dt = _target_dt_per_residue(
        target,
        model_input,
        dtype=prediction.translation_local.dtype,
        device=prediction.translation_local.device,
    )
    if dt is None:
        dt = prediction.translation_local.new_full(
            (target.target_valid.shape[0],), 1.0 / float(fixed_steps)
        )
    target_fraction = (dt / graph_time.clamp_min(dt)).clamp(max=1.0)
    prediction_fraction = (
        dt if prediction_type in ("velocity", "bridge_velocity")
        else target_fraction
    )
    predicted_next = predict_holo_coordinates(
        model_input,
        _scale_prediction(
            prediction,
            prediction_fraction,
            backbone_only=backbone_only,
        ),
        apply_chi=not backbone_only,
    )
    if prediction_type == "bridge_velocity":
        if target.next_protein_pos is None:
            raise ValueError("bridge_velocity requires target.next_protein_pos")
        oracle_next = target.next_protein_pos.detach()
    else:
        oracle_next = predict_holo_coordinates(
            model_input,
            _scale_prediction(
                DiffusionMotionPrediction(
                    target.translation_target_local,
                    target.rotation_target_local,
                    target.chi_target,
                ),
                target_fraction,
                backbone_only=backbone_only,
            ),
            apply_chi=not backbone_only,
        ).detach()
    valid_residue = target.target_valid & model_input.frame_valid
    atom_valid = valid_residue[model_input.atom_to_residue]
    if backbone_only:
        atom_valid = atom_valid & _backbone_atom_mask(model_input)
    if not bool(atom_valid.any()):
        return predicted_next.sum() * 0.0
    return (predicted_next[atom_valid] - oracle_next[atom_valid]).square().mean()


def diffusion_motion_loss(prediction: DiffusionMotionPrediction,
                           target: DiffusionStateTarget,
                           *,
                           model_input: Optional[DiffusionStateInput] = None,
                           endpoint_weight: float = 0.0,
                           score_normalize: bool = False,
                           score_floor: float = 0.1,
                           motion_parameterization: str = "remaining",
                           prediction_type: str = "velocity",
                           chi_loss_weight: float = 1.0,
                           next_state_weight: float = 0.0,
                           next_state_steps: int = 20,
                           backbone_only: bool = False,
                           backbone_endpoint_weight: float = 0.0,
                           continuity_weight: float = 0.0,
                           direction_weight: float = 0.0,
                           direction_threshold: float = 0.05,
                           backbone_progress_direction_weight: float = 0.0,
                           backbone_progress_magnitude_weight: float = 0.0,
                           backbone_progress_threshold: float = 0.05) -> DiffusionMotionLoss:
    """Return motion targets plus an optional coordinate endpoint objective.

    ``model_input`` is optional for backwards compatibility with the original
    local SE(3)+χ loss.  Supplying it computes ``endpoint_loss`` even when its
    weight is zero, which lets training diagnostics record the coordinate
    calibration term without changing optimization.
    """
    if endpoint_weight < 0.0 or not torch.isfinite(torch.tensor(endpoint_weight)):
        raise ValueError("endpoint_weight must be a finite non-negative scalar")
    if score_floor <= 0.0 or score_floor > 1.0 or not torch.isfinite(torch.tensor(score_floor)):
        raise ValueError("score_floor must be finite and lie in (0, 1]")
    if motion_parameterization not in ("remaining", "bridge_rate"):
        raise ValueError(
            "motion_parameterization must be remaining or bridge_rate"
        )
    if prediction_type not in ("remaining", "velocity", "bridge_velocity"):
        raise ValueError("unsupported prediction_type")
    if (
        backbone_endpoint_weight < 0.0
        or continuity_weight < 0.0
        or direction_weight < 0.0
        or backbone_progress_direction_weight < 0.0
        or backbone_progress_magnitude_weight < 0.0
        or chi_loss_weight < 0.0
        or next_state_weight < 0.0
    ):
        raise ValueError("endpoint auxiliary weights must be non-negative")
    if direction_threshold < 0.0 or not torch.isfinite(
        torch.tensor(direction_threshold)
    ):
        raise ValueError("direction_threshold must be finite and non-negative")
    if backbone_progress_threshold < 0.0 or not torch.isfinite(
        torch.tensor(backbone_progress_threshold)
    ):
        raise ValueError(
            "backbone_progress_threshold must be finite and non-negative"
        )
    valid = target.target_valid
    if not bool(valid.any()):
        raise ValueError("diffusion target has no valid residues")
    if model_input is not None:
        residue_count = int(target.target_valid.shape[0])
        if model_input.residue_type.shape[0] != residue_count:
            raise ValueError("model_input and target residue counts differ")
        batch_residue = model_input.batch_residue
        if batch_residue is None:
            batch_residue = torch.zeros(
                residue_count,
                dtype=torch.long,
                device=target.target_valid.device,
            )
        if batch_residue.shape != (residue_count,) or batch_residue.dtype != torch.long:
            raise ValueError("batch_residue must be LongTensor [num_residues]")
        diffusion_time = model_input.diffusion_time.reshape(-1).to(
            dtype=prediction.translation_local.dtype,
            device=prediction.translation_local.device,
        )
        if diffusion_time.numel() == 0 or not bool(torch.isfinite(diffusion_time).all()):
            raise ValueError("model_input.diffusion_time must be finite and non-empty")
        if batch_residue.numel() and int(batch_residue.max()) >= diffusion_time.numel():
            raise ValueError("batch_residue references a missing diffusion time")
        graph_time = diffusion_time[batch_residue.to(device=diffusion_time.device)]
    else:
        graph_time = prediction.translation_local.new_ones(
            target.target_valid.shape[0]
        )

    if motion_parameterization == "bridge_rate":
        if model_input is None:
            raise ValueError(
                "model_input is required when motion_parameterization is bridge_rate"
            )
        bridge_time = graph_time.clamp(0.05, 1.0)
        translation_target = target.translation_target_local / bridge_time[:, None]
        rotation_target = target.rotation_target_local / bridge_time[:, None]
        chi_target = target.chi_target / bridge_time[:, None]
    else:
        bridge_time = prediction.translation_local.new_ones(
            target.target_valid.shape[0]
        )
        translation_target = target.translation_target_local
        rotation_target = target.rotation_target_local
        chi_target = target.chi_target
    if prediction_type == "velocity":
        if model_input is None:
            raise ValueError("model_input is required for velocity targets")
        # t=1 is the apo boundary. The conditional flow target is the
        # current-to-holo transform per unit reverse-time interval.
        flow_time = graph_time.clamp_min(0.02)
        translation_target = translation_target / flow_time[:, None]
        rotation_target = rotation_target / flow_time[:, None]
        chi_target = chi_target / flow_time[:, None]
    elif prediction_type == "bridge_velocity":
        if target.target_dt is None:
            raise ValueError("bridge_velocity requires target.target_dt")
    if score_normalize:
        if model_input is None:
            raise ValueError("model_input is required when score_normalize is enabled")
        # The state noise is proportional to t.  Keep a non-zero floor so
        # clean-endpoint examples constrain the score without exploding.
        sigma = score_floor + (1.0 - score_floor) * graph_time.clamp(0.0, 1.0)
    else:
        sigma = prediction.translation_local.new_ones(
            target.target_valid.shape[0]
        )
    translation_residual = (
        prediction.translation_local[valid] - translation_target[valid]
    )
    rotation_residual = (
        prediction.rotation_local[valid] - rotation_target[valid]
    )
    translation_loss = (translation_residual / sigma[valid, None]).square().mean()
    rotation_loss = (rotation_residual / sigma[valid, None]).square().mean()
    chi_valid = target.target_valid[:, None] & torch.isfinite(chi_target)
    chi_valid = chi_valid & (chi_target.abs() > 0.0)
    if bool(chi_valid.any()):
        chi_residual = torch.atan2(
            torch.sin(prediction.chi[chi_valid] - chi_target[chi_valid]),
            torch.cos(prediction.chi[chi_valid] - chi_target[chi_valid]),
        )
        if score_normalize:
            chi_sigma = sigma[:, None].expand_as(chi_target)[chi_valid]
            chi_loss = (chi_residual / chi_sigma).square().mean()
        else:
            chi_loss = (1.0 - torch.cos(chi_residual)).mean()
    else:
        chi_loss = prediction.chi.sum() * 0.0
    if backbone_only:
        chi_loss = prediction.chi.sum() * 0.0
    direction_mask = valid & (
        torch.linalg.vector_norm(translation_target, dim=-1)
        > float(direction_threshold)
    )
    if bool(direction_mask.any()):
        target_norm = torch.linalg.vector_norm(
            translation_target[direction_mask], dim=-1
        ).clamp_min(1.0e-6)
        predicted_norm = torch.linalg.vector_norm(
            prediction.translation_local[direction_mask], dim=-1
        )
        target_direction = translation_target[direction_mask] / target_norm[:, None]
        predicted_direction = prediction.translation_local[direction_mask] / (
            predicted_norm.square()[:, None] + 1.0e-4
        ).sqrt()
        direction_loss = (
            1.0 - (target_direction * predicted_direction).sum(dim=-1)
        ).mean()
    else:
        direction_loss = prediction.translation_local.sum() * 0.0
    if model_input is None:
        if endpoint_weight != 0.0:
            raise ValueError("model_input is required when endpoint_weight is non-zero")
        endpoint_loss = prediction.translation_local.sum() * 0.0
    else:
        endpoint_prediction = prediction
        if prediction_type == "bridge_velocity":
            endpoint_loss = prediction.translation_local.sum() * 0.0
        else:
            if prediction_type == "velocity":
                flow_time = graph_time.clamp_min(0.02)
                endpoint_prediction = DiffusionMotionPrediction(
                    translation_local=prediction.translation_local * flow_time[:, None],
                    rotation_local=prediction.rotation_local * flow_time[:, None],
                    chi=prediction.chi * flow_time[:, None],
                )
            endpoint_loss = diffusion_endpoint_loss(
                endpoint_prediction, target, model_input, backbone_only=backbone_only
            )
    if motion_parameterization == "bridge_rate" and model_input is not None:
        # The bridge-rate head predicts a unit-time update.  Endpoint
        # supervision must reconstruct the full remaining transform at the
        # sampled time before applying it to the current coordinates.
        endpoint_prediction = DiffusionMotionPrediction(
            translation_local=prediction.translation_local * bridge_time[:, None],
            rotation_local=prediction.rotation_local * bridge_time[:, None],
            chi=prediction.chi * bridge_time[:, None],
        )
        endpoint_loss = diffusion_endpoint_loss(
            endpoint_prediction,
            target,
            model_input,
            backbone_only=backbone_only,
        )
    backbone_loss = prediction.translation_local.sum() * 0.0
    next_state_loss = prediction.translation_local.sum() * 0.0
    continuity_loss = prediction.translation_local.sum() * 0.0
    backbone_progress_direction_loss = prediction.translation_local.sum() * 0.0
    backbone_progress_magnitude_loss = prediction.translation_local.sum() * 0.0
    if model_input is not None and (
        backbone_endpoint_weight > 0.0
        or continuity_weight > 0.0
        or next_state_weight > 0.0
        or backbone_progress_direction_weight > 0.0
        or backbone_progress_magnitude_weight > 0.0
    ):
        backbone_mask = _backbone_atom_mask(model_input)
        valid_atom_mask = target.target_valid[model_input.atom_to_residue]
        valid_atom_mask = valid_atom_mask & model_input.frame_valid[
            model_input.atom_to_residue
        ]
        backbone_mask = backbone_mask & valid_atom_mask
        endpoint_prediction = prediction
        if prediction_type == "bridge_velocity":
            if target.next_protein_pos is None:
                raise ValueError("bridge_velocity requires target.next_protein_pos")
            target_dt = _target_dt_per_residue(
                target,
                model_input,
                dtype=prediction.translation_local.dtype,
                device=prediction.translation_local.device,
            )
            if target_dt is None:
                target_dt = prediction.translation_local.new_full(
                    (prediction.translation_local.shape[0],),
                    1.0 / float(next_state_steps),
                )
            predicted_aux = predict_holo_coordinates(
                model_input,
                _scale_prediction(
                    prediction,
                    target_dt,
                    backbone_only=backbone_only,
                ),
                apply_chi=not backbone_only,
            )
            auxiliary_target = target.next_protein_pos.detach()
        else:
            if prediction_type == "velocity":
                flow_time = graph_time.clamp_min(0.02)
                endpoint_prediction = DiffusionMotionPrediction(
                    translation_local=prediction.translation_local * flow_time[:, None],
                    rotation_local=prediction.rotation_local * flow_time[:, None],
                    chi=prediction.chi * flow_time[:, None],
                )
            predicted_aux = predict_holo_coordinates(
                model_input, endpoint_prediction, apply_chi=not backbone_only
            )
            auxiliary_target = target.protein_pos_holo
        if bool(backbone_mask.any()):
            backbone_loss = (
                predicted_aux[backbone_mask] - auxiliary_target[backbone_mask]
            ).square().mean()
        if (
            backbone_progress_direction_weight > 0.0
            or backbone_progress_magnitude_weight > 0.0
        ):
            (
                backbone_progress_direction_loss,
                backbone_progress_magnitude_loss,
            ) = _backbone_progress_loss(
                predicted_aux,
                auxiliary_target,
                model_input,
                target,
                threshold=backbone_progress_threshold,
            )
        continuity_loss = _chain_continuity_loss(
            predicted_aux, auxiliary_target, model_input
        )
        if next_state_weight > 0.0:
            next_state_loss = diffusion_next_state_loss(
                prediction,
                target,
                model_input,
                prediction_type=prediction_type,
                fixed_steps=next_state_steps,
                backbone_only=backbone_only,
            )
    total = (
        translation_loss
        + rotation_loss
        + float(chi_loss_weight) * chi_loss
        + float(endpoint_weight) * endpoint_loss
        + float(backbone_endpoint_weight) * backbone_loss
        + float(next_state_weight) * next_state_loss
        + float(continuity_weight) * continuity_loss
        + float(direction_weight) * direction_loss
        + float(backbone_progress_direction_weight)
        * backbone_progress_direction_loss
        + float(backbone_progress_magnitude_weight)
        * backbone_progress_magnitude_loss
    )
    return DiffusionMotionLoss(
        total,
        translation_loss,
        rotation_loss,
        chi_loss,
        endpoint_loss,
        backbone_loss,
        next_state_loss,
        continuity_loss,
        direction_loss,
        backbone_progress_direction_loss,
        backbone_progress_magnitude_loss,
    )


__all__ = [
    "DiffusionMotionAdapter",
    "DiffusionMotionLoss",
    "DiffusionMotionPrediction",
    "predict_holo_coordinates",
    "diffusion_endpoint_loss",
    "diffusion_motion_loss",
]

@dataclass(frozen=True)
class ReverseTrajectory:
    states: tuple[torch.Tensor, ...]
    times: tuple[float, ...]


def sample_reverse_trajectory(model: DiffusionMotionAdapter, state, *, steps: int = 8,
                              step_scale: float = 1.0,
                              remaining_step_offset: float = 0.0,
                              motion_parameterization: str = "remaining",
                              prediction_type: str = "velocity",
                              use_chi: bool = True,
                              ligand_conditioner=None,
                              ligand_generator: Optional[torch.Generator] = None) -> ReverseTrajectory:
    """Run a deterministic continuous reverse trajectory from apo to holo.

    ``prediction_type="velocity"`` is the v4.5 conditional-flow contract:
    the network predicts d(state)/dt for the normalized path t=1 (apo) to
    t=0 (holo), and Euler integrates exactly ``steps`` intervals. The older
    remaining-transform path is retained only for checkpoint compatibility.
    """
    if steps <= 0 or step_scale <= 0:
        raise ValueError("steps and step_scale must be positive")
    if (
        remaining_step_offset < 0.0
        or not torch.isfinite(torch.tensor(remaining_step_offset))
    ):
        raise ValueError("remaining_step_offset must be finite and non-negative")
    if motion_parameterization not in ("remaining", "bridge_rate"):
        raise ValueError(
            "motion_parameterization must be remaining or bridge_rate"
        )
    if prediction_type not in ("remaining", "velocity", "bridge_velocity"):
        raise ValueError("unsupported prediction_type")
    current = state.model_input.protein_pos.detach().clone()
    current_chi = state.model_input.chi_current.detach().clone()
    # Topology is fixed throughout a trajectory. Reusing it avoids rebuilding
    # Python-side residue metadata and re-extracting every dihedral twice per
    # step; the coordinate update itself remains the same.
    chi_metadata = build_chi_update_metadata(
        state.model_input.protein_atom_name,
        state.model_input.atom_to_residue,
        state.model_input.protein_residue_name,
    )
    # Phase 44 t=1 is apo-side.  Build a fresh input object per step while
    # preserving all label-independent fields and replacing only coordinates,
    # χ and continuous time.
    from dataclasses import replace
    states = [current.clone()]
    times = [1.0]
    for index in range(steps):
        # Condition on the state actually passed to the network.  The first
        # state is apo (t=1); the update then advances it to the next time.
        # Using the next time here shifts every rollout one step away from
        # the training state contract.
        current_t = 1.0 - float(index) / float(steps)
        next_t = 1.0 - float(index + 1) / float(steps)
        inp = replace(state.model_input, protein_pos=current,
                      chi_current=current_chi,
                      diffusion_time=torch.tensor(
                          [current_t], dtype=current.dtype, device=current.device
                      ))
        if ligand_conditioner is not None:
            condition = ligand_conditioner.condition_from_input(
                inp,
                current_t,
                generator=ligand_generator,
            )
            inp = replace(
                inp,
                ligand_pos=condition.ligand_pos.to(
                    device=current.device, dtype=torch.float32
                ),
                ligand_type=condition.ligand_type.to(
                    device=current.device, dtype=torch.long
                ),
                ligand_condition_t=torch.tensor(
                    [condition.targetdiff_t],
                    dtype=torch.long,
                    device=current.device,
                ),
            )
        with torch.no_grad():
            prediction = model(inp)
        frames = build_residue_frames(
            current,
            inp.atom_to_residue,
            inp.protein_atom_name,
            num_residues=inp.residue_type.shape[0],
            atom_indices=inp.frame_atom_indices,
        )
        remaining_steps = steps - index
        if prediction_type in ("velocity", "bridge_velocity"):
            fraction = step_scale / float(steps)
        elif motion_parameterization == "bridge_rate":
            fraction = step_scale / (steps + remaining_step_offset)
        else:
            fraction = step_scale / (remaining_steps + remaining_step_offset)
        current = apply_local_se3_update(
            current,
            inp.atom_to_residue,
            frames.origins,
            frames.frames,
            prediction.translation_local,
            prediction.rotation_local,
            # The prediction is a remaining transform to the holo endpoint.
            # Its fraction must therefore use the number of updates left, not
            # the original trajectory length; otherwise even an oracle model
            # systematically undershoots the endpoint.
            fraction=fraction,
            frame_valid=inp.frame_valid & frames.valid,
        )
        if use_chi:
            chi_valid = (
                inp.chi_mask
                & chi_metadata.valid
                & inp.frame_valid[:, None]
            )
            chi_update = apply_chi_updates(
                current,
                chi_metadata.axis_start,
                chi_metadata.axis_end,
                chi_metadata.downstream_atom_mask,
                prediction.chi * fraction,
                valid=chi_valid,
            )
            current = chi_update.positions
            current_chi = torch.atan2(
                torch.sin(current_chi + chi_update.applied_chi),
                torch.cos(current_chi + chi_update.applied_chi),
            )
        states.append(current.clone())
        times.append(next_t)
    return ReverseTrajectory(tuple(states), tuple(times))
