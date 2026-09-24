"""Remaining-transform labels, fractional residue updates, and oracle bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from .so3 import so3_exp, so3_log


Tensor = torch.Tensor


@dataclass(frozen=True)
class RemainingTransform:
    """Current-residue-local transform needed to reach the holo endpoint."""

    translation_local: Tensor
    rotvec_local: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if self.translation_local.ndim != 2 or self.translation_local.shape[-1] != 3:
            raise ValueError("translation_local must have shape [Nr, 3]")
        if self.rotvec_local.shape != self.translation_local.shape:
            raise ValueError("rotvec_local and translation_local must have identical shapes")
        if self.valid.dtype != torch.bool or self.valid.shape != self.translation_local.shape[:1]:
            raise ValueError("valid must be BoolTensor [Nr]")
        if not torch.isfinite(self.translation_local).all() or not torch.isfinite(self.rotvec_local).all():
            raise ValueError("remaining transform must be finite")


@dataclass(frozen=True)
class BridgeState:
    """One synthetic bridge state ``P_k`` in the apo coordinate frame."""

    protein_pos: Tensor
    origins: Tensor
    frames: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if self.protein_pos.ndim != 2 or self.protein_pos.shape[-1] != 3:
            raise ValueError("protein_pos must have shape [N, 3]")
        if self.origins.ndim != 2 or self.origins.shape[-1] != 3:
            raise ValueError("origins must have shape [Nr, 3]")
        if self.frames.ndim != 3 or self.frames.shape[-2:] != (3, 3):
            raise ValueError("frames must have shape [Nr, 3, 3]")
        if self.frames.shape[0] != self.origins.shape[0]:
            raise ValueError("frames and origins must have identical residue counts")
        if self.valid.dtype != torch.bool or self.valid.shape != self.origins.shape[:1]:
            raise ValueError("valid must be BoolTensor [Nr]")
        if not torch.isfinite(self.protein_pos).all() or not torch.isfinite(self.origins).all() or not torch.isfinite(self.frames).all():
            raise ValueError("bridge state must be finite")


@dataclass(frozen=True)
class OracleReconstructionMetrics:
    """Minimal Phase 2 oracle report."""

    atom_rmsd: float
    backbone_rmsd: float
    frame_invalid_count: int
    valid_atom_count: int
    valid_backbone_atom_count: int


def _check_residue_geometry(
    origins: Tensor,
    frames: Tensor,
    other_origins: Tensor,
    other_frames: Tensor,
    valid: Optional[Tensor],
) -> Tensor:
    for name, value in (("origins", origins), ("other_origins", other_origins)):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [Nr, 3]")
    for name, value in (("frames", frames), ("other_frames", other_frames)):
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-2:] != (3, 3):
            raise ValueError(f"{name} must have shape [Nr, 3, 3]")
    if origins.shape != other_origins.shape or frames.shape != other_frames.shape or frames.shape[0] != origins.shape[0]:
        raise ValueError("current and holo residue geometry shapes must match")
    if valid is None:
        valid = torch.ones(origins.shape[0], dtype=torch.bool, device=origins.device)
    if valid.dtype != torch.bool or valid.shape != origins.shape[:1]:
        raise ValueError("valid must be BoolTensor [Nr]")
    return valid


def remaining_transform_current_to_holo(
    current_origins: Tensor,
    current_frames: Tensor,
    holo_origins: Tensor,
    holo_frames: Tensor,
    *,
    frame_valid: Optional[Tensor] = None,
) -> RemainingTransform:
    """Compute current-local translation and rotation toward holo."""

    valid = _check_residue_geometry(current_origins, current_frames, holo_origins, holo_frames, frame_valid)
    current_origins = current_origins.to(dtype=torch.float32)
    current_frames = current_frames.to(dtype=torch.float32)
    holo_origins = holo_origins.to(dtype=torch.float32)
    holo_frames = holo_frames.to(dtype=torch.float32)
    rotation_local = current_frames.transpose(-1, -2) @ holo_frames
    rotvec_local = so3_log(rotation_local)
    translation_local = torch.bmm(
        (holo_origins - current_origins).unsqueeze(1), current_frames
    ).squeeze(1)
    translation_local = torch.where(valid[:, None], translation_local, torch.zeros_like(translation_local))
    rotvec_local = torch.where(valid[:, None], rotvec_local, torch.zeros_like(rotvec_local))
    return RemainingTransform(translation_local=translation_local, rotvec_local=rotvec_local, valid=valid)


def apply_fractional_update(
    protein_pos: Tensor,
    atom_to_residue: Tensor,
    current_origins: Tensor,
    current_frames: Tensor,
    remaining_translation_local: Tensor,
    remaining_rotvec_local: Tensor,
    *,
    remaining_steps: Union[int, Tensor],
    frame_valid: Optional[Tensor] = None,
) -> Tensor:
    """Apply one ``1 / remaining_steps`` local SE(3) update to residue atoms.

    ``remaining_steps`` may be a positive scalar or one positive value per
    residue, which lets a batched solver use independent per-graph schedules.
    """

    if not isinstance(protein_pos, torch.Tensor) or protein_pos.ndim != 2 or protein_pos.shape[-1] != 3:
        raise ValueError("protein_pos must have shape [N, 3]")
    if not protein_pos.is_floating_point():
        raise TypeError("protein_pos must use a floating dtype")
    if atom_to_residue.dtype != torch.long or atom_to_residue.ndim != 1 or atom_to_residue.shape[0] != protein_pos.shape[0]:
        raise ValueError("atom_to_residue must be LongTensor [N]")
    if remaining_translation_local.shape != remaining_rotvec_local.shape or remaining_translation_local.ndim != 2 or remaining_translation_local.shape[-1] != 3:
        raise ValueError("remaining local transforms must have shape [Nr, 3]")
    if current_origins.shape != remaining_translation_local.shape or current_frames.shape != (remaining_translation_local.shape[0], 3, 3):
        raise ValueError("current residue geometry and remaining transforms have incompatible shapes")
    if isinstance(remaining_steps, torch.Tensor):
        if remaining_steps.numel() == 1:
            steps = remaining_steps.to(device=protein_pos.device, dtype=torch.float32).reshape(1)
        elif remaining_steps.ndim == 1 and remaining_steps.shape[0] == current_origins.shape[0]:
            steps = remaining_steps.to(device=protein_pos.device, dtype=torch.float32)
        else:
            raise ValueError("remaining_steps must be a scalar or one value per residue")
    else:
        steps = torch.tensor(float(remaining_steps), dtype=torch.float32, device=protein_pos.device).reshape(1)
    if not torch.isfinite(steps).all() or bool((steps <= 0.0).any()):
        raise ValueError("remaining_steps must be positive and finite")
    if frame_valid is None:
        frame_valid = torch.ones(current_origins.shape[0], dtype=torch.bool, device=current_origins.device)
    if frame_valid.dtype != torch.bool or frame_valid.shape != current_origins.shape[:1]:
        raise ValueError("frame_valid must be BoolTensor [Nr]")
    if atom_to_residue.numel() and (int(atom_to_residue.min()) < 0 or int(atom_to_residue.max()) >= current_origins.shape[0]):
        raise ValueError("atom_to_residue contains an out-of-range residue id")

    positions = protein_pos.to(dtype=torch.float32)
    origins = current_origins.to(dtype=torch.float32)
    frames = current_frames.to(dtype=torch.float32)
    translation_local = remaining_translation_local.to(dtype=torch.float32) / steps.reshape(-1, 1)
    rotvec_local = remaining_rotvec_local.to(dtype=torch.float32) / steps.reshape(-1, 1)
    step_rotation_local = so3_exp(rotvec_local)
    step_rotation_global = frames @ step_rotation_local @ frames.transpose(-1, -2)
    step_translation_global = torch.bmm(
        translation_local.unsqueeze(1), frames.transpose(-1, -2)
    ).squeeze(1)

    atom_origins = origins[atom_to_residue]
    atom_rotation = step_rotation_global[atom_to_residue]
    atom_translation = step_translation_global[atom_to_residue]
    centered = (positions - atom_origins).unsqueeze(1)
    updated = torch.bmm(centered, atom_rotation.transpose(-1, -2)).squeeze(1) + atom_origins + atom_translation
    valid_atoms = frame_valid[atom_to_residue]
    zero_residue_update = (
        remaining_translation_local == 0.0
    ).all(dim=-1) & (remaining_rotvec_local == 0.0).all(dim=-1)
    updated = torch.where(zero_residue_update[atom_to_residue][:, None], positions, updated)
    return torch.where(valid_atoms[:, None], updated, positions)


def build_bridge_state(
    apo_pos: Tensor,
    atom_to_residue: Tensor,
    apo_origins: Tensor,
    apo_frames: Tensor,
    holo_origins: Tensor,
    holo_frames: Tensor,
    *,
    fraction: Union[float, Tensor],
    frame_valid: Optional[Tensor] = None,
) -> BridgeState:
    """Construct the deterministic synthetic bridge state at ``s∈[0,1]``."""

    valid = _check_residue_geometry(apo_origins, apo_frames, holo_origins, holo_frames, frame_valid)
    if not isinstance(apo_pos, torch.Tensor) or apo_pos.ndim != 2 or apo_pos.shape[-1] != 3:
        raise ValueError("apo_pos must have shape [N, 3]")
    if atom_to_residue.dtype != torch.long or atom_to_residue.ndim != 1 or atom_to_residue.shape[0] != apo_pos.shape[0]:
        raise ValueError("atom_to_residue must be LongTensor [N]")
    if atom_to_residue.numel() and (int(atom_to_residue.min()) < 0 or int(atom_to_residue.max()) >= apo_origins.shape[0]):
        raise ValueError("atom_to_residue contains an out-of-range residue id")
    fraction_tensor = torch.as_tensor(fraction, dtype=torch.float32, device=apo_pos.device)
    if fraction_tensor.numel() not in (1, apo_origins.shape[0]):
        raise ValueError("fraction must be scalar or one value per residue")
    fraction_tensor = fraction_tensor.reshape(-1)
    if not torch.isfinite(fraction_tensor).all() or bool((fraction_tensor < 0.0).any()) or bool((fraction_tensor > 1.0).any()):
        raise ValueError("fraction must lie in [0, 1]")
    if fraction_tensor.numel() == 1:
        fraction_tensor = fraction_tensor.expand(apo_origins.shape[0])

    apo_pos = apo_pos.to(dtype=torch.float32)
    apo_origins = apo_origins.to(dtype=torch.float32)
    apo_frames = apo_frames.to(dtype=torch.float32)
    holo_origins = holo_origins.to(dtype=torch.float32)
    holo_frames = holo_frames.to(dtype=torch.float32)
    relative_global = holo_frames @ apo_frames.transpose(-1, -2)
    relative_rotvec = so3_log(relative_global)
    interpolation_rotation = so3_exp(fraction_tensor[:, None] * relative_rotvec)
    bridge_frames = interpolation_rotation @ apo_frames
    bridge_origins = (1.0 - fraction_tensor[:, None]) * apo_origins + fraction_tensor[:, None] * holo_origins

    atom_rotation = interpolation_rotation[atom_to_residue]
    atom_apo_origin = apo_origins[atom_to_residue]
    atom_bridge_origin = bridge_origins[atom_to_residue]
    reconstructed = torch.bmm(
        (apo_pos - atom_apo_origin).unsqueeze(1), atom_rotation.transpose(-1, -2)
    ).squeeze(1) + atom_bridge_origin
    reconstructed = torch.where(valid[atom_to_residue][:, None], reconstructed, apo_pos)
    bridge_frames = torch.where(valid[:, None, None], bridge_frames, apo_frames)
    bridge_origins = torch.where(valid[:, None], bridge_origins, apo_origins)
    return BridgeState(
        protein_pos=reconstructed,
        origins=bridge_origins,
        frames=bridge_frames,
        valid=valid,
    )


def oracle_reconstruction_metrics(
    reconstructed_pos: Tensor,
    holo_pos: Tensor,
    atom_to_residue: Tensor,
    atom_name: Sequence[str],
    frame_valid: Tensor,
) -> OracleReconstructionMetrics:
    """Compute masked atom/backbone RMSD and invalid-frame count."""

    if reconstructed_pos.shape != holo_pos.shape or reconstructed_pos.ndim != 2 or reconstructed_pos.shape[-1] != 3:
        raise ValueError("reconstructed_pos and holo_pos must both have shape [N, 3]")
    if atom_to_residue.dtype != torch.long or atom_to_residue.shape != reconstructed_pos.shape[:1]:
        raise ValueError("atom_to_residue must be LongTensor [N]")
    if frame_valid.dtype != torch.bool or atom_to_residue.numel() and int(atom_to_residue.max()) >= frame_valid.shape[0]:
        raise ValueError("frame_valid is incompatible with atom_to_residue")
    valid_atom = frame_valid[atom_to_residue]
    error_sq = (reconstructed_pos.to(torch.float32) - holo_pos.to(torch.float32)).square().sum(dim=-1)
    valid_count = int(valid_atom.sum())
    atom_rmsd = float(torch.sqrt(error_sq[valid_atom].mean()).item()) if valid_count else float("nan")
    backbone_mask = valid_atom & torch.tensor(
        [name in {"N", "CA", "C", "O"} for name in atom_name], dtype=torch.bool, device=valid_atom.device
    )
    backbone_count = int(backbone_mask.sum())
    backbone_rmsd = float(torch.sqrt(error_sq[backbone_mask].mean()).item()) if backbone_count else float("nan")
    return OracleReconstructionMetrics(
        atom_rmsd=atom_rmsd,
        backbone_rmsd=backbone_rmsd,
        frame_invalid_count=int((~frame_valid).sum()),
        valid_atom_count=valid_count,
        valid_backbone_atom_count=backbone_count,
    )


__all__ = [
    "BridgeState",
    "OracleReconstructionMetrics",
    "RemainingTransform",
    "apply_fractional_update",
    "build_bridge_state",
    "oracle_reconstruction_metrics",
    "remaining_transform_current_to_holo",
]
