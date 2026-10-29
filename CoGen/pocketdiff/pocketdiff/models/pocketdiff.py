"""Independent PocketDiff MVP forward model.

The model predicts residue-local remaining motion. Coordinate updates are
separate: legacy rigid updates live in geometry.bridge; the explicit current-χ
model uses the independent differentiable geometry.joint_update path.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from pocketdiff.data.schema import PocketComplex, PocketDiffPrediction
from pocketdiff.data.apo2mol_adapter import AA_NAMES
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.chi import ChiUpdateMetadata
from pocketdiff.geometry.frames import ResidueFrameResult, build_residue_frames
from pocketdiff.geometry.so3 import so3_log

from .chi_head import ResidueChiHead
from .invariant_encoder import DistanceInvariantEncoder
from .motion_head import ResidueMotionHead
from .residue_pool import scatter_mean_residue
from .time_embedding import SinusoidalTimeEmbedding
from .targetdiff_encoder import TargetDiffEncoderAdapter


class PocketDiffModel(nn.Module):
    """Independent apo→holo residue motion predictor.

    Rigid motion remains the historical default.  ``predict_chi=True`` adds a
    separate periodic side-chain head without changing the frozen 283-d rigid
    descriptor or the state-dict shape of older models.
    Set ``chi_input_mode="current"`` for the independent joint-step path: χ is
    recomputed from current coordinates and stored under distinct head keys.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        encoder_layers: int = 2,
        knn: int = 32,
        num_rbf: int = 16,
        sigma_translation: float = 1.0,
        dropout: float = 0.1,
        motion_parameterization: str = "remaining",
        predict_chi: bool = False,
        chi_input_mode: str = "apo",
        encoder_backend: str = "scalar",
        stage_gate: bool = False,
        t_switch_ratio: float = 0.4,
    ) -> None:
        super().__init__()
        if motion_parameterization not in ("remaining", "bridge_rate"):
            raise ValueError("motion_parameterization must be remaining or bridge_rate")
        self.motion_parameterization = motion_parameterization
        if encoder_backend not in ("scalar", "targetdiff"):
            raise ValueError("encoder_backend must be scalar or targetdiff")
        self.encoder_backend = encoder_backend
        self.predict_chi = bool(predict_chi)
        if chi_input_mode not in ("apo", "current"):
            raise ValueError("chi_input_mode must be apo or current")
        if chi_input_mode == "current" and not predict_chi:
            raise ValueError("current chi mode requires predict_chi=True")
        self.chi_input_mode = chi_input_mode
        if not 0.0 < float(t_switch_ratio) < 1.0:
            raise ValueError("t_switch_ratio must lie in (0, 1)")
        self.stage_gate = bool(stage_gate)
        self.t_switch_ratio = float(t_switch_ratio)
        mode_contract = "pocketdiff-current-joint-v1" if chi_input_mode == "current" else "pocketdiff-legacy-apo-v1"
        self.input_contract = mode_contract + ":encoder=" + encoder_backend
        if hidden_dim != 128:
            raise ValueError("the frozen MVP descriptor contract requires hidden_dim=128")
        self.hidden_dim = hidden_dim
        self.time_embedding = SinusoidalTimeEmbedding(64)
        self.pocket_time_embedding = SinusoidalTimeEmbedding(64)
        self.protein_time_proj = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.ligand_time_proj = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        nn.init.zeros_(self.protein_time_proj[-1].weight)
        nn.init.zeros_(self.protein_time_proj[-1].bias)
        nn.init.zeros_(self.ligand_time_proj[-1].weight)
        nn.init.zeros_(self.ligand_time_proj[-1].bias)
        if encoder_backend == "scalar":
            self.encoder = DistanceInvariantEncoder(
                hidden_dim=hidden_dim,
                num_layers=encoder_layers,
                knn=knn,
                num_rbf=num_rbf,
            )
        else:
            self.encoder = TargetDiffEncoderAdapter()
        self.motion_head = ResidueMotionHead(sigma_translation=sigma_translation, dropout=dropout)
        if self.predict_chi:
            if self.chi_input_mode == "current":
                # Different parameter keys deliberately prevent a legacy apo-chi
                # checkpoint from silently acquiring current-state semantics.
                self.current_chi_head = ResidueChiHead(dropout=dropout)
            else:
                self.chi_head = ResidueChiHead(dropout=dropout)

    @staticmethod
    def _validate_batch(
        protein_pos: torch.Tensor,
        apo_pos_ref: torch.Tensor,
        protein_feature: torch.Tensor,
        atom_to_residue_global: torch.Tensor,
        residue_type: torch.Tensor,
        frame_valid: torch.Tensor,
        batch_protein: torch.Tensor,
        batch_residue: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        targetdiff_t: torch.Tensor,
        pocket_k: torch.Tensor,
        protein_atom_name: Sequence[str],
    ) -> int:
        if protein_pos.ndim != 2 or protein_pos.shape[-1] != 3 or apo_pos_ref.shape != protein_pos.shape:
            raise ValueError("protein_pos and apo_pos_ref must both have shape [Np, 3]")
        if protein_feature.shape != (protein_pos.shape[0], 27):
            raise ValueError("protein_feature must have shape [Np, 27]")
        if atom_to_residue_global.dtype != torch.long or atom_to_residue_global.shape != protein_pos.shape[:1]:
            raise ValueError("atom_to_residue_global must be LongTensor [Np]")
        num_residues = int(residue_type.shape[0])
        if residue_type.dtype != torch.long or residue_type.ndim != 1 or num_residues == 0:
            raise ValueError("residue_type must be non-empty LongTensor [Nr]")
        if int(residue_type.min()) < 0 or int(residue_type.max()) >= 20:
            raise ValueError("residue_type must lie in [0, 19]")
        if frame_valid.dtype != torch.bool or frame_valid.shape != (num_residues,):
            raise ValueError("frame_valid must be BoolTensor [Nr]")
        if len(protein_atom_name) != protein_pos.shape[0]:
            raise ValueError("protein_atom_name must contain one name per protein atom")
        if batch_protein.dtype != torch.long or batch_protein.shape != protein_pos.shape[:1] or batch_protein.numel() == 0:
            raise ValueError("batch_protein must be non-empty LongTensor [Np]")
        if batch_residue.dtype != torch.long or batch_residue.shape != (num_residues,):
            raise ValueError("batch_residue must be LongTensor [Nr]")
        if batch_protein.min() < 0 or batch_residue.min() < 0:
            raise ValueError("batch ids cannot be negative")
        batch_size = int(batch_protein.max().item()) + 1
        if int(batch_residue.max().item()) + 1 != batch_size:
            raise ValueError("batch_residue must cover the same graphs as batch_protein")
        if atom_to_residue_global.numel() and (int(atom_to_residue_global.min()) < 0 or int(atom_to_residue_global.max()) >= num_residues):
            raise ValueError("atom_to_residue_global contains an out-of-range residue id")
        if not torch.equal(batch_residue[atom_to_residue_global], batch_protein):
            raise ValueError("atom_to_residue_global crosses graph boundaries")
        if ligand_pos.ndim != 2 or ligand_pos.shape[-1] != 3:
            raise ValueError("ligand_pos must have shape [Nl, 3]")
        if ligand_v.dtype != torch.long or ligand_v.shape != (ligand_pos.shape[0],):
            raise ValueError("ligand_v must be LongTensor [Nl]")
        if ligand_v.numel() and (int(ligand_v.min()) < 0 or int(ligand_v.max()) >= 13):
            raise ValueError("ligand_v must lie in [0, 12]")
        if batch_ligand.dtype != torch.long or batch_ligand.shape != (ligand_pos.shape[0],):
            raise ValueError("batch_ligand must be LongTensor [Nl]")
        if batch_ligand.numel() and (int(batch_ligand.min()) < 0 or int(batch_ligand.max()) >= batch_size):
            raise ValueError("batch_ligand contains an out-of-range graph id")
        if targetdiff_t.dtype != torch.long or targetdiff_t.shape != (batch_size,):
            raise ValueError("targetdiff_t must be LongTensor [B]")
        if pocket_k.dtype != torch.long or pocket_k.shape != (batch_size,):
            raise ValueError("pocket_k must be LongTensor [B]")
        if targetdiff_t.numel() and (int(targetdiff_t.min()) < 0 or int(targetdiff_t.max()) > 999):
            raise ValueError("targetdiff_t must lie in [0, 999]")
        if pocket_k.numel() and (int(pocket_k.min()) < 0 or int(pocket_k.max()) > 19):
            raise ValueError("pocket_k must lie in [0, 19]")
        return batch_size

    def forward(
        self,
        protein_pos: torch.Tensor,
        apo_pos_ref: torch.Tensor,
        protein_feature: torch.Tensor,
        atom_to_residue_global: torch.Tensor,
        residue_type: torch.Tensor,
        frame_valid: torch.Tensor,
        batch_protein: torch.Tensor,
        batch_residue: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        targetdiff_t: torch.Tensor,
        pocket_k: torch.Tensor,
        *,
        protein_atom_name: Optional[Sequence[str]] = None,
        chi_apo: Optional[torch.Tensor] = None,
        chi_mask: Optional[torch.Tensor] = None,
        frame_atom_indices: Optional[torch.Tensor] = None,
        chi_metadata: Optional[ChiUpdateMetadata] = None,
        apo_frame: Optional[ResidueFrameResult] = None,
    ) -> PocketDiffPrediction:
        if protein_atom_name is None:
            raise ValueError("protein_atom_name is required to construct N–CA–C frames")
        if self.chi_input_mode == "current" and (chi_apo is not None or chi_mask is not None):
            raise ValueError("current chi mode rejects legacy chi_apo/chi_mask inputs")
        batch_size = self._validate_batch(
            protein_pos,
            apo_pos_ref,
            protein_feature,
            atom_to_residue_global,
            residue_type,
            frame_valid,
            batch_protein,
            batch_residue,
            ligand_pos,
            ligand_v,
            batch_ligand,
            targetdiff_t,
            pocket_k,
            protein_atom_name,
        )
        current_frames = build_residue_frames(
            protein_pos,
            atom_to_residue_global,
            protein_atom_name,
            num_residues=residue_type.shape[0],
            atom_indices=frame_atom_indices,
        )
        if apo_frame is None:
            apo_frames = build_residue_frames(
                apo_pos_ref,
                atom_to_residue_global,
                protein_atom_name,
                num_residues=residue_type.shape[0],
                atom_indices=frame_atom_indices,
            )
        else:
            if not isinstance(apo_frame, ResidueFrameResult):
                raise TypeError("apo_frame must be a ResidueFrameResult")
            if (
                apo_frame.num_residues != residue_type.shape[0]
                or apo_frame.origins.device != protein_pos.device
                or apo_frame.frames.device != protein_pos.device
                or apo_frame.valid.device != protein_pos.device
            ):
                raise ValueError("apo_frame is incompatible with the model batch")
            apo_frames = apo_frame
        valid = frame_valid & current_frames.valid & apo_frames.valid
        target_time = self.time_embedding(targetdiff_t.to(dtype=torch.float32) / 999.0)
        pocket_time = self.pocket_time_embedding(pocket_k.to(dtype=torch.float32) / 20.0)
        combined_time = torch.cat((target_time, pocket_time), dim=-1)
        protein_time = self.protein_time_proj(combined_time)
        ligand_time = self.ligand_time_proj(combined_time)
        if self.encoder_backend == "scalar":
            protein_h, _ = self.encoder(
                protein_pos, protein_feature, batch_protein, ligand_pos, ligand_v,
                batch_ligand, protein_time, ligand_time,
            )
        else:
            encoded = self.encoder(
                protein_pos, protein_feature, batch_protein, ligand_pos, ligand_v, batch_ligand
            )
            protein_h = encoded["protein_hidden"]
        residue_h = scatter_mean_residue(protein_h, atom_to_residue_global, residue_type.shape[0])
        current_to_apo_translation = torch.bmm(
            (apo_frames.origins - current_frames.origins).unsqueeze(1), current_frames.frames
        ).squeeze(1)
        current_to_apo_rotation = so3_log(current_frames.frames.transpose(-1, -2) @ apo_frames.frames)
        current_to_apo_translation = torch.where(valid[:, None], current_to_apo_translation, torch.zeros_like(current_to_apo_translation))
        current_to_apo_rotation = torch.where(valid[:, None], current_to_apo_rotation, torch.zeros_like(current_to_apo_rotation))
        residue_one_hot = F.one_hot(residue_type, num_classes=20).to(dtype=torch.float32)
        valid_float = valid.to(dtype=torch.float32)[:, None]
        target_time_residue = target_time[batch_residue]
        pocket_time_residue = pocket_time[batch_residue]
        descriptor = torch.cat(
            (
                residue_h,
                current_to_apo_translation,
                current_to_apo_rotation,
                residue_one_hot,
                valid_float,
                target_time_residue,
                pocket_time_residue,
            ),
            dim=-1,
        )
        translation, rotvec = self.motion_head(descriptor)
        if self.motion_parameterization == "bridge_rate":
            # Public outputs keep actual remaining units. The rate head assumes
            # the ideal bridge schedule; autonomous states need separate validation.
            fraction = (20 - pocket_k[batch_residue]).to(translation.dtype)[:, None] / 20
            translation = translation * fraction
            rotvec = rotvec * fraction
        translation = torch.where(valid[:, None], translation, torch.zeros_like(translation))
        rotvec = torch.where(valid[:, None], rotvec, torch.zeros_like(rotvec))
        remaining_chi = None
        if self.chi_input_mode == "current":
            state = build_current_chi_state(
                protein_pos, protein_atom_name, atom_to_residue_global,
                [AA_NAMES[i] for i in residue_type.detach().cpu().tolist()],
                metadata=chi_metadata,
            )
            chi_mask = state.geometry_rotatable_mask
            remaining_chi = self.current_chi_head(descriptor, state.angles, chi_mask)
            remaining_chi = torch.where(valid[:, None], remaining_chi, torch.zeros_like(remaining_chi))
        elif self.predict_chi:
            if chi_apo is None or chi_mask is None:
                raise ValueError("chi_apo and chi_mask are required when predict_chi=True")
            if not chi_apo.is_floating_point():
                raise TypeError("chi_apo must use a floating dtype")
            if chi_apo.shape != (residue_type.shape[0], 5):
                raise ValueError("chi_apo must have shape [Nr, 5]")
            if chi_mask.dtype != torch.bool or chi_mask.shape != chi_apo.shape:
                raise ValueError("chi_mask must be BoolTensor with shape [Nr, 5]")
            if not torch.isfinite(chi_apo).all():
                raise ValueError("chi_apo must be finite")
            remaining_chi = self.chi_head(descriptor, chi_apo, chi_mask)
            remaining_chi = torch.where(valid[:, None], remaining_chi,
                                        torch.zeros_like(remaining_chi))
        if self.stage_gate:
            stage_start = int(round(self.t_switch_ratio * 20.0))
            active = pocket_k[batch_residue] >= stage_start
            translation = torch.where(active[:, None], translation, torch.zeros_like(translation))
            rotvec = torch.where(active[:, None], rotvec, torch.zeros_like(rotvec))
            if remaining_chi is not None:
                remaining_chi = torch.where(active[:, None], remaining_chi, torch.zeros_like(remaining_chi))
        if remaining_chi is not None and self.motion_parameterization == "bridge_rate":
            # The joint solver always divides a public remaining motion by the
            # number of steps left.  In bridge-rate mode the head instead
            # emits the total twenty-step rate, so all three heads (rigid and
            # periodic chi) are converted to the public units uniformly.
            fraction = (20 - pocket_k[batch_residue]).to(remaining_chi.dtype)[:, None] / 20
            remaining_chi = remaining_chi * fraction
        diagnostics: Dict[str, torch.Tensor] = {
            "valid_frame_fraction": valid.to(dtype=torch.float32).mean(),
            "residue_hidden_norm": torch.linalg.vector_norm(residue_h, dim=-1).mean(),
            "descriptor_norm": torch.linalg.vector_norm(descriptor, dim=-1).mean(),
        }
        if remaining_chi is not None:
            diagnostics["valid_chi_fraction"] = (
                (chi_mask & valid[:, None]).to(dtype=torch.float32).mean()
            )
            diagnostics["chi_norm"] = torch.linalg.vector_norm(remaining_chi, dim=-1).mean()
        return PocketDiffPrediction(
            remaining_translation_local=translation,
            remaining_rotvec_local=rotvec,
            remaining_chi=remaining_chi,
            frame_valid=valid,
            diagnostics=diagnostics,
        )

    def forward_complex(
        self,
        complex_value: PocketComplex,
        *,
        targetdiff_t: Union[int, torch.Tensor] = 199,
        pocket_k: Union[int, torch.Tensor] = 0,
        protein_pos: Optional[torch.Tensor] = None,
    ) -> PocketDiffPrediction:
        """Convenience single-graph call for a Phase 1 ``PocketComplex``."""

        device = complex_value.protein_pos_apo.device
        pos = complex_value.protein_pos_apo if protein_pos is None else protein_pos
        t = torch.as_tensor(targetdiff_t, dtype=torch.long, device=device).reshape(1)
        k = torch.as_tensor(pocket_k, dtype=torch.long, device=device).reshape(1)
        zeros_protein = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
        zeros_residue = torch.zeros(complex_value.num_residues, dtype=torch.long, device=device)
        zeros_ligand = torch.zeros(complex_value.num_ligand_atoms, dtype=torch.long, device=device)
        if self.chi_input_mode == "current":
            frame_valid = build_residue_frames(
                complex_value.protein_pos_apo, complex_value.atom_to_residue,
                complex_value.protein_atom_name, num_residues=complex_value.num_residues,
            ).valid
            chi_kwargs = {}
        else:
            frame_valid = complex_value.frame_valid.to(device=device)
            chi_kwargs = dict(chi_apo=complex_value.chi_apo.to(device=device),
                              chi_mask=complex_value.chi_mask.to(device=device))
        return self.forward(
            pos,
            complex_value.protein_pos_apo.to(device=device),
            complex_value.protein_feature.to(device=device),
            complex_value.atom_to_residue.to(device=device),
            complex_value.residue_type.to(device=device),
            frame_valid,
            zeros_protein,
            zeros_residue,
            complex_value.ligand_pos_ref.to(device=device),
            complex_value.ligand_type_ref.to(device=device),
            zeros_ligand,
            t,
            k,
            protein_atom_name=complex_value.protein_atom_name,
            **chi_kwargs,
        )


__all__ = ["PocketDiffModel"]
