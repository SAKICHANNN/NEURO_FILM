"""Private learned RGB-node predictor; explicit trilinear final rendering."""

import torch
from torch import nn
from torch.nn import functional as F

from .lut import apply_lut, identity_lut


class ConditionedRGBLUT(nn.Module):
    def __init__(self, conditional=True, size=9):
        super().__init__()
        self.conditional, self.size = conditional, size
        self.register_buffer("identity", identity_lut(size), persistent=False)
        self.global_residual = nn.Parameter(torch.zeros(3, size, size, size, 3))
        if conditional:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.embedding = nn.Embedding(3, 8)
            self.head = nn.Sequential(
                nn.Linear(72, 64), nn.ReLU(), nn.Linear(64, 3 * size**3)
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def predict_lut(self, x, style):
        if x.ndim != 4 or x.shape[1] != 3 or not torch.isfinite(x).all():
            raise ValueError("Expected finite BCHW RGB")
        if (x < 0).any() or (x > 1).any():
            raise ValueError("Expected unit RGB")
        if style.shape != (len(x),) or (style < 0).any() or (style > 2).any():
            raise ValueError("Expected valid style IDs")
        residual = self.global_residual[style]
        if self.conditional:
            context = F.interpolate(x, (64, 64), mode="bilinear", align_corners=False)
            features = torch.cat([self.encoder(context), self.embedding(style)], 1)
            residual = residual + self.head(features).reshape(
                -1, self.size, self.size, self.size, 3
            )
        return (self.identity + residual).clamp(0, 1)

    def forward(self, x, style):
        return apply_lut(x, self.predict_lut(x, style))
