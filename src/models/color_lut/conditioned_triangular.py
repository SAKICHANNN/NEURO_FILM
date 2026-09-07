"""Private image-conditioned explicit operator, not a neural RGB generator."""

import torch
from torch import nn
from torch.nn import functional as F


def apply_parameters(x, raw):
    if x.ndim != 4 or x.shape[1] != 3 or raw.shape != (len(x), 3, 12):
        raise ValueError("Expected BCHW RGB and Bx3x12 parameters")
    if not torch.isfinite(x).all() or not torch.isfinite(raw).all():
        raise ValueError("Nonfinite input")
    if (x < 0).any() or (x > 1).any():
        raise ValueError("Expected unit RGB")
    p = raw.tanh()
    centres = torch.linspace(-4, 4, 9, device=x.device, dtype=x.dtype)
    rows = []
    for c in range(3):
        xc = x[:, c : c + 1]
        endpoint = (xc == 0) | (xc == 1)
        z = torch.logit(torch.where(endpoint, torch.full_like(xc, 0.5), xc))
        delta = 2 * p[:, c, 0, None, None, None]
        for k in range(9):
            delta = delta + (0.5 / 9) * p[:, c, k + 1, None, None, None] * torch.tanh(
                z - centres[k]
            )
        for j in range(c):
            delta = delta + 0.5 * p[:, c, 10 + j, None, None, None] * (
                x[:, j : j + 1] - 0.5
            )
        rows.append(torch.where(endpoint, xc, torch.sigmoid(z + delta)))
    return torch.cat(rows, 1)


class ConditionedTriangular(nn.Module):
    def __init__(self, conditional=True):
        super().__init__()
        self.conditional = conditional
        self.global_parameters = nn.Parameter(torch.zeros(3, 3, 12))
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
            self.head = nn.Sequential(nn.Linear(72, 64), nn.ReLU(), nn.Linear(64, 36))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def predict_parameters(self, x, style):
        if style.shape != (len(x),) or (style < 0).any() or (style > 2).any():
            raise ValueError("Expected one valid style ID per image")
        base = self.global_parameters[style]
        if not self.conditional:
            return base
        context = F.interpolate(x, (64, 64), mode="bilinear", align_corners=False)
        features = torch.cat([self.encoder(context), self.embedding(style)], 1)
        return base + self.head(features).reshape(-1, 3, 12)

    def forward(self, x, style):
        return apply_parameters(x, self.predict_parameters(x, style))
