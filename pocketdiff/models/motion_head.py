"""Residue-local remaining SE(3) head."""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidueMotionHead(nn.Module):
    """Residue-local SE(3) head with a configurable output initialization."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 283,
        sigma_translation: float = 1.0,
        dropout: float = 0.1,
        zero_init: bool = True,
        output_std: float = 0.005,
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        if sigma_translation <= 0.0:
            raise ValueError("sigma_translation must be positive")
        if output_std <= 0.0:
            raise ValueError("output_std must be positive")
        self.sigma_translation = float(sigma_translation)
        self.descriptor_dim = int(descriptor_dim)
        self.network = nn.Sequential(
            nn.Linear(self.descriptor_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 6),
        )
        if zero_init:
            nn.init.zeros_(self.network[-1].weight)
        else:
            # DynamicBind factorizes vector direction and magnitude and does
            # not start with a completely blocked output path.  A small
            # random final layer keeps that gradient path alive without
            # creating Å-scale updates at initialization.
            nn.init.normal_(self.network[-1].weight, mean=0.0, std=output_std)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, descriptor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptor.ndim != 2 or descriptor.shape[-1] != self.descriptor_dim:
            raise ValueError(
                f"descriptor must have shape [Nr, {self.descriptor_dim}]"
            )
        raw = self.network(descriptor)
        translation = raw[:, :3] * self.sigma_translation
        raw_rotvec = raw[:, 3:]
        norm = torch.linalg.vector_norm(raw_rotvec, dim=-1, keepdim=True)
        # Use the analytic limit tanh(||r||)/||r|| → 1 at the origin.  A plain
        # ``norm.clamp_min(eps)`` would make the zero-initialized head receive
        # zero rotation gradient forever.
        safe_norm = norm.clamp_min(1.0e-8)
        scale = torch.where(
            norm > 1.0e-8,
            torch.tanh(norm) / safe_norm,
            torch.ones_like(norm),
        )
        rotvec = math.pi * scale * raw_rotvec
        return translation, rotvec


__all__ = ["ResidueMotionHead"]
