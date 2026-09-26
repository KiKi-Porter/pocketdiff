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
    continuity_loss: torch.Tensor
    direction_loss: torch.Tensor


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


def _chain_continuity_loss(
    predicted: torch.Tensor,
    holo: torch.Tensor,
    model_input: DiffusionStateInput,
) -> torch.Tensor:
    """Penalize CA and peptide C-N geometry drift at the predicted endpoint."""

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
    if not distances_pred:
        return predicted.sum() * 0.0
    return torch.stack(
        [(a - b).square() for a, b in zip(distances_pred, distances_holo)]
    ).mean()


class DiffusionMotionAdapter(nn.Module):
    """Residue diffusion adapter with scalar, TargetDiff, and vector backends."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        encoder_backend: str = "scalar",
        prediction_type: str = "velocity",
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
        if prediction_type not in ("remaining", "velocity"):
            raise ValueError("prediction_type must be remaining or velocity")
        self.prediction_type = prediction_type
        if encoder_backend in ("scalar", "dynamicbind"):
            self.encoder = DistanceInvariantEncoder(hidden_dim=hidden_dim)
            cross_num_rbf = self.encoder.num_rbf
        else:
            self.encoder = TargetDiffEncoderAdapter()
            cross_num_rbf = self.encoder.config["num_rbf"]
        self.descriptor_dim = 289
        self.motion_head = ResidueMotionHead(
            descriptor_dim=self.descriptor_dim,
            dropout=dropout,
            zero_init=encoder_backend != "dynamicbind",
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
        residue_h = scatter_mean_residue(protein_h, value.atom_to_residue, value.residue_type.shape[0])
        residue_one_hot = torch.nn.functional.one_hot(value.residue_type, num_classes=20).float()
        current_frames = build_residue_frames(
            pos,
            value.atom_to_residue,
            value.protein_atom_name,
            num_residues=value.residue_type.shape[0],
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
        descriptor = torch.cat((residue_h, residue_one_hot, value.frame_valid[:, None].float(),
                                time_h[batch_residue],
                                directional_h, sequence_h), dim=-1)
        if descriptor.shape[-1] != self.descriptor_dim:
            raise RuntimeError(
                "DiffusionMotionAdapter descriptor contract changed: "
                f"expected {self.descriptor_dim}, got {descriptor.shape[-1]}"
            )
        translation, rotation = self.motion_head(descriptor)
        chi = self.chi_head(descriptor, value.chi_current, value.chi_mask)
        return DiffusionMotionPrediction(translation, rotation, chi)


def predict_holo_coordinates(
    model_input: DiffusionStateInput,
    prediction: DiffusionMotionPrediction,
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
) -> torch.Tensor:
    """Measure the predicted one-update coordinate endpoint against holo.

    The residue supervision mask is expanded to atoms through
    ``atom_to_residue``.  Invalid residues contribute no coordinate error, so
    missing/degenerate frame records cannot create a false endpoint signal.
    """
    predicted_holo = predict_holo_coordinates(model_input, prediction)
    valid_residue = target.target_valid & model_input.frame_valid
    atom_valid = valid_residue[model_input.atom_to_residue]
    if not bool(atom_valid.any()):
        return predicted_holo.sum() * 0.0
    return (
        predicted_holo[atom_valid] - target.protein_pos_holo[atom_valid]
    ).square().mean()


def diffusion_motion_loss(prediction: DiffusionMotionPrediction,
                           target: DiffusionStateTarget,
                           *,
                           model_input: Optional[DiffusionStateInput] = None,
                           endpoint_weight: float = 0.0,
                           score_normalize: bool = False,
                           score_floor: float = 0.1,
                           motion_parameterization: str = "remaining",
                           prediction_type: str = "velocity",
                           backbone_endpoint_weight: float = 0.0,
                           continuity_weight: float = 0.0,
                           direction_weight: float = 0.0,
                           direction_threshold: float = 0.05) -> DiffusionMotionLoss:
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
    if prediction_type not in ("remaining", "velocity"):
        raise ValueError("prediction_type must be remaining or velocity")
    if (
        backbone_endpoint_weight < 0.0
        or continuity_weight < 0.0
        or direction_weight < 0.0
    ):
        raise ValueError("endpoint auxiliary weights must be non-negative")
    if direction_threshold < 0.0 or not torch.isfinite(
        torch.tensor(direction_threshold)
    ):
        raise ValueError("direction_threshold must be finite and non-negative")
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
        if prediction_type == "velocity":
            flow_time = graph_time.clamp_min(0.02)
            endpoint_prediction = DiffusionMotionPrediction(
                translation_local=prediction.translation_local * flow_time[:, None],
                rotation_local=prediction.rotation_local * flow_time[:, None],
                chi=prediction.chi * flow_time[:, None],
            )
        endpoint_loss = diffusion_endpoint_loss(
            endpoint_prediction, target, model_input
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
        )
    backbone_loss = prediction.translation_local.sum() * 0.0
    continuity_loss = prediction.translation_local.sum() * 0.0
    if model_input is not None and (backbone_endpoint_weight > 0.0 or continuity_weight > 0.0):
        backbone_mask = _backbone_atom_mask(model_input)
        endpoint_prediction = prediction
        if prediction_type == "velocity":
            flow_time = graph_time.clamp_min(0.02)
            endpoint_prediction = DiffusionMotionPrediction(
                translation_local=prediction.translation_local * flow_time[:, None],
                rotation_local=prediction.rotation_local * flow_time[:, None],
                chi=prediction.chi * flow_time[:, None],
            )
        predicted_holo = predict_holo_coordinates(model_input, endpoint_prediction)
        if bool(backbone_mask.any()):
            backbone_loss = (
                predicted_holo[backbone_mask] - target.protein_pos_holo[backbone_mask]
            ).square().mean()
        continuity_loss = _chain_continuity_loss(
            predicted_holo, target.protein_pos_holo, model_input
        )
    total = (
        translation_loss
        + rotation_loss
        + chi_loss
        + float(endpoint_weight) * endpoint_loss
        + float(backbone_endpoint_weight) * backbone_loss
        + float(continuity_weight) * continuity_loss
        + float(direction_weight) * direction_loss
    )
    return DiffusionMotionLoss(
        total,
        translation_loss,
        rotation_loss,
        chi_loss,
        endpoint_loss,
        backbone_loss,
        continuity_loss,
        direction_loss,
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
    if prediction_type not in ("remaining", "velocity"):
        raise ValueError("prediction_type must be remaining or velocity")
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
        )
        remaining_steps = steps - index
        if prediction_type == "velocity":
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
