"""Small sinusoidal time embeddings used by the independent MVP."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("time embedding dim must be a positive even number")
        self.dim = dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 1:
            raise ValueError("time values must have shape [B]")
        values = values.to(dtype=torch.float32)
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, dtype=values.dtype, device=values.device
        ) / max(half - 1, 1)
        frequencies = exponent.exp()
        phase = values[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


__all__ = ["SinusoidalTimeEmbedding"]
