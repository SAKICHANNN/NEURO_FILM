"""Differentiable 3D LUT operators for content-preserving color rendering."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def identity_lut(size: int = 33, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return an RGB identity LUT with shape ``[size, size, size, 3]``."""
    axis = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    r, g, b = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([r, g, b], dim=-1)


def _apply_single_lut(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be [B, 3, H, W], got {tuple(image.shape)}")
    if lut.ndim != 4 or lut.shape[-1] != 3 or len(set(lut.shape[:3])) != 1:
        raise ValueError(f"lut must be [D, D, D, 3], got {tuple(lut.shape)}")

    batch, _, height, width = image.shape
    size = lut.shape[0]
    coords = image.clamp(0.0, 1.0).permute(0, 2, 3, 1) * (size - 1)
    lower = coords.floor().long()
    upper = (lower + 1).clamp(max=size - 1)
    frac = coords - lower.to(coords.dtype)

    r0, g0, b0 = lower.unbind(dim=-1)
    r1, g1, b1 = upper.unbind(dim=-1)
    wr, wg, wb = frac.unbind(dim=-1)
    flat = lut.reshape(size * size * size, 3)

    def gather(ri: torch.Tensor, gi: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
        index = ri * size * size + gi * size + bi
        return flat[index.reshape(-1)].reshape(batch, height, width, 3)

    c000 = gather(r0, g0, b0)
    c001 = gather(r0, g0, b1)
    c010 = gather(r0, g1, b0)
    c011 = gather(r0, g1, b1)
    c100 = gather(r1, g0, b0)
    c101 = gather(r1, g0, b1)
    c110 = gather(r1, g1, b0)
    c111 = gather(r1, g1, b1)

    wr = wr.unsqueeze(-1)
    wg = wg.unsqueeze(-1)
    wb = wb.unsqueeze(-1)
    c00 = c000 * (1.0 - wb) + c001 * wb
    c01 = c010 * (1.0 - wb) + c011 * wb
    c10 = c100 * (1.0 - wb) + c101 * wb
    c11 = c110 * (1.0 - wb) + c111 * wb
    c0 = c00 * (1.0 - wg) + c01 * wg
    c1 = c10 * (1.0 - wg) + c11 * wg
    out = c0 * (1.0 - wr) + c1 * wr
    return out.permute(0, 3, 1, 2).contiguous()


def apply_lut(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply a single or per-image 3D LUT to an RGB tensor.

    Args:
        image: Tensor shaped ``[B, 3, H, W]`` in ``[0, 1]``.
        lut: Tensor shaped ``[D, D, D, 3]`` or ``[B, D, D, D, 3]``.
    """
    if lut.ndim == 4:
        return _apply_single_lut(image, lut)
    if lut.ndim != 5 or lut.shape[0] != image.shape[0]:
        raise ValueError(f"batch LUT must be [B, D, D, D, 3], got {tuple(lut.shape)}")
    outputs = [_apply_single_lut(image[index : index + 1], lut[index]) for index in range(image.shape[0])]
    return torch.cat(outputs, dim=0)


class BasisLUT(nn.Module):
    """Identity LUT plus learnable basis residuals."""

    def __init__(self, size: int = 33, num_basis: int = 4, residual_scale: float = 0.25) -> None:
        super().__init__()
        self.size = size
        self.num_basis = num_basis
        self.residual_scale = residual_scale
        self.register_buffer("identity", identity_lut(size), persistent=False)
        self.basis = nn.Parameter(torch.zeros(num_basis, size, size, size, 3))

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim != 2 or weights.shape[1] != self.num_basis:
            raise ValueError(f"weights must be [B, {self.num_basis}], got {tuple(weights.shape)}")
        residual = torch.einsum("bk,k...c->b...c", weights, self.basis)
        return (self.identity.unsqueeze(0) + residual * self.residual_scale).clamp(0.0, 1.0)


class TinyLUTEncoder(nn.Module):
    """Small image encoder that predicts basis LUT weights."""

    def __init__(self, num_basis: int = 4, style_count: int = 8, style_dim: int = 16) -> None:
        super().__init__()
        self.style_embedding = nn.Embedding(style_count, style_dim)
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + style_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_basis),
        )

    def forward(self, image: torch.Tensor, style_index: torch.Tensor) -> torch.Tensor:
        pooled = self.features(image).flatten(1)
        style = self.style_embedding(style_index)
        logits = self.head(torch.cat([pooled, style], dim=1))
        return F.softmax(logits, dim=1)
