"""Style-separated SepLUT model for neural film rendering research."""

from __future__ import annotations

import torch
from torch import nn

from src.models.color_lut import apply_lut, identity_lut


STYLE_NAMES = [
    "ektar_100",
    "portra_400",
    "portra_800",
    "velvia_50",
    "vision3_250d",
    "vision3_500t",
]


def identity_1d_lut(size: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    return axis[None, :].repeat(3, 1)


def apply_1d_lut(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply per-channel 1D LUTs.

    Args:
        image: `[B, 3, H, W]` in `[0, 1]`.
        lut: `[3, N]` or `[B, 3, N]`.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be [B, 3, H, W], got {tuple(image.shape)}")
    if lut.ndim == 2:
        lut = lut.unsqueeze(0).expand(image.shape[0], -1, -1)
    if lut.ndim != 3 or lut.shape[0] != image.shape[0] or lut.shape[1] != 3:
        raise ValueError(f"lut must be [3, N] or [B, 3, N], got {tuple(lut.shape)}")

    batch, _channels, height, width = image.shape
    size = lut.shape[-1]
    coords = image.clamp(0.0, 1.0) * (size - 1)
    lower = coords.floor().long()
    upper = (lower + 1).clamp(max=size - 1)
    frac = coords - lower.to(coords.dtype)
    out = torch.empty_like(image)
    for channel in range(3):
        channel_lut = lut[:, channel, :]
        flat_lut = channel_lut.reshape(batch * size)
        offset = torch.arange(batch, device=image.device).view(batch, 1, 1) * size
        lo = flat_lut[(lower[:, channel] + offset).reshape(-1)].reshape(batch, height, width)
        hi = flat_lut[(upper[:, channel] + offset).reshape(-1)].reshape(batch, height, width)
        out[:, channel] = lo * (1.0 - frac[:, channel]) + hi * frac[:, channel]
    return out


class ImageStyleGate(nn.Module):
    """Small image/style encoder that modulates per-style LUT strength."""

    def __init__(self, style_count: int, style_dim: int = 16) -> None:
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
            nn.Linear(64, 2),
        )

    def forward(self, image: torch.Tensor, style_index: torch.Tensor) -> torch.Tensor:
        pooled = self.features(image).flatten(1)
        style = self.style_embedding(style_index)
        raw = self.head(torch.cat([pooled, style], dim=1))
        return 0.70 + 0.60 * torch.sigmoid(raw)


class StyleSeparatedSepLUT(nn.Module):
    """Per-stock 1D + 3D LUT renderer with image-specific gates.

    This avoids the V1 shared-softmax basis collapse by giving each stock its
    own residual LUT parameters. Image specificity enters through lightweight
    gates that modulate tone/channel and color-volume residuals.
    """

    def __init__(
        self,
        *,
        style_count: int = len(STYLE_NAMES),
        lut1d_size: int = 33,
        lut3d_size: int = 17,
        residual_1d_scale: float = 0.35,
        residual_3d_scale: float = 0.22,
    ) -> None:
        super().__init__()
        self.style_count = style_count
        self.lut1d_size = lut1d_size
        self.lut3d_size = lut3d_size
        self.residual_1d_scale = residual_1d_scale
        self.residual_3d_scale = residual_3d_scale
        self.register_buffer("identity_1d", identity_1d_lut(lut1d_size), persistent=False)
        self.register_buffer("identity_3d", identity_lut(lut3d_size), persistent=False)
        self.residual_1d = nn.Parameter(torch.zeros(style_count, 3, lut1d_size))
        self.residual_3d = nn.Parameter(torch.zeros(style_count, lut3d_size, lut3d_size, lut3d_size, 3))
        self.gate = ImageStyleGate(style_count)

    def lut_1d(self, style_index: torch.Tensor, strength: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.residual_1d[style_index]) * self.residual_1d_scale
        scale = strength[:, None, None] * gate[:, 0:1, None]
        lut = self.identity_1d.unsqueeze(0) + residual * scale
        return lut.clamp(0.0, 1.0)

    def lut_3d(self, style_index: torch.Tensor, strength: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.residual_3d[style_index]) * self.residual_3d_scale
        scale = strength[:, None, None, None, None] * gate[:, 1:2, None, None, None]
        lut = self.identity_3d.unsqueeze(0) + residual * scale
        return lut.clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor, style_index: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        if strength.ndim == 0:
            strength = strength.expand(image.shape[0])
        strength = strength.to(device=image.device, dtype=image.dtype).clamp(0.0, 2.0)
        style_index = style_index.to(device=image.device, dtype=torch.long)
        gate = self.gate(image, style_index)
        after_1d = apply_1d_lut(image, self.lut_1d(style_index, strength, gate))
        after_3d = apply_lut(after_1d, self.lut_3d(style_index, strength, gate))
        return after_3d.clamp(0.0, 1.0)

    def regularization(self) -> dict[str, torch.Tensor]:
        one_d = self.identity_1d.unsqueeze(0) + torch.tanh(self.residual_1d) * self.residual_1d_scale
        diffs = one_d[:, :, 1:] - one_d[:, :, :-1]
        monotonic = torch.relu(-diffs).mean()
        smooth_1d = (one_d[:, :, 2:] - 2.0 * one_d[:, :, 1:-1] + one_d[:, :, :-2]).abs().mean()
        residual_3d = torch.tanh(self.residual_3d) * self.residual_3d_scale
        smooth_3d = (
            (residual_3d[:, 1:] - residual_3d[:, :-1]).abs().mean()
            + (residual_3d[:, :, 1:] - residual_3d[:, :, :-1]).abs().mean()
            + (residual_3d[:, :, :, 1:] - residual_3d[:, :, :, :-1]).abs().mean()
        ) / 3.0
        return {
            "monotonic_1d": monotonic,
            "smooth_1d": smooth_1d,
            "smooth_3d": smooth_3d,
        }
