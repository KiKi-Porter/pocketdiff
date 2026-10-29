"""Residue-local periodic side-chain χ head."""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidueChiHead(nn.Module):
    """Predict five bounded remaining χ angles from a residue descriptor.

    The caller supplies explicitly apo or current χ sine/cosine features.
    Separate model parameter namespaces preserve these input semantics. Input is
    continuous at the periodic boundary.  The final layer is zero initialized
    to preserve the staged model's zero-update starting point, while the
    analytic ``tanh`` path keeps gradients finite at initialization.
    """

    def __init__(
        self,
        *,
        descriptor_dim: int = 283,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        zero_init: bool = True,
        output_std: float = 0.005,
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0 or hidden_dim <= 0:
            raise ValueError("descriptor_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if output_std <= 0.0:
            raise ValueError("output_std must be positive")
        self.descriptor_dim = int(descriptor_dim)
        self.input_dim = self.descriptor_dim + 15  # sin χ, cos χ, valid mask
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 5),
        )
        if zero_init:
            nn.init.zeros_(self.network[-1].weight)
        else:
            nn.init.normal_(self.network[-1].weight, mean=0.0, std=output_std)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        descriptor: torch.Tensor,
        chi_current: torch.Tensor,
        chi_mask: torch.Tensor,
    ) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.shape[-1] != self.descriptor_dim:
            raise ValueError(
                f"descriptor must have shape [Nr, {self.descriptor_dim}]"
            )
        if chi_current.ndim != 2 or chi_current.shape != (descriptor.shape[0], 5):
            raise ValueError("chi_current must have shape [Nr, 5]")
        if chi_mask.dtype != torch.bool or chi_mask.shape != chi_current.shape:
            raise ValueError("chi_mask must be BoolTensor with shape [Nr, 5]")
        if not torch.isfinite(chi_current).all():
            raise ValueError("chi_current must be finite")
        chi_features = torch.cat(
            (torch.sin(chi_current), torch.cos(chi_current),
             chi_mask.to(dtype=descriptor.dtype)),
            dim=-1,
        )
        raw = self.network(torch.cat((descriptor, chi_features), dim=-1))
        # The output is an actual angle delta, not a normalized bridge rate.
        # tanh bounds it to the principal periodic interval and has a nonzero
        # derivative at zero.
        output = math.pi * torch.tanh(raw)
        return torch.where(chi_mask, output, torch.zeros_like(output))


__all__ = ["ResidueChiHead"]
