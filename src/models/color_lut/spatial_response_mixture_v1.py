import torch
from torch import nn
from torch.nn import functional as F

from .lut import apply_lut, identity_lut


class SpatialResponseMixture(nn.Module):
    def __init__(self, spatial: bool = True, size: int = 9, bases: int = 4, styles: int = 3, initialization_noise: float = 0.001):
        super().__init__()
        self.spatial, self.size, self.bases = spatial, size, bases
        self.register_buffer("identity", identity_lut(size), persistent=False)
        noise = torch.randn(styles, bases, size, size, size, 3) * initialization_noise
        noise -= noise.mean(1, keepdim=True)
        self.basis_residual = nn.Parameter(noise)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.embedding = nn.Embedding(styles, 8)
        self.head = nn.Sequential(nn.Conv2d(72, 64, 1), nn.ReLU(), nn.Conv2d(64, bases, 1))
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def predict_weights(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(F.interpolate(x, (64, 64), mode="bilinear", align_corners=False))
        if not self.spatial:
            feature = feature.mean((-1, -2), keepdim=True)
        embedding = self.embedding(style)[:, :, None, None].expand(-1, -1, *feature.shape[-2:])
        return self.head(torch.cat((feature, embedding), 1)).softmax(1)

    def predict_bases(self, style: torch.Tensor) -> torch.Tensor:
        return (self.identity[None, None] + self.basis_residual[style]).clamp(0, 1)

    def render(self, x: torch.Tensor, bases: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        dense = F.interpolate(weights, x.shape[-2:], mode="bilinear", align_corners=False)
        dense = dense / dense.sum(1, keepdim=True)
        result = sum(apply_lut(x, bases[:, index]) * dense[:, index:index + 1] for index in range(self.bases))
        return result.clamp(0, 1)

    def forward(self, x: torch.Tensor, style: torch.Tensor, misalign: bool = False) -> torch.Tensor:
        weights = self.predict_weights(x, style)
        if misalign:
            weights = weights.roll((weights.shape[-2] // 2, weights.shape[-1] // 2), (-2, -1))
        return self.render(x, self.predict_bases(style), weights)


def effective_probe_luts(bases: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    probes = F.adaptive_avg_pool2d(weights, (2, 2)).flatten(2).transpose(1, 2)
    return torch.einsum("bpk,bkijlc->bpijlc", probes, bases).flatten(0, 1)


def spatial_weight_penalty(weights: torch.Tensor) -> torch.Tensor:
    terms = [weights.diff(dim=axis).square().mean() for axis in (-1, -2) if weights.shape[axis] > 1]
    return torch.stack(terms).mean() if terms else weights.sum() * 0
