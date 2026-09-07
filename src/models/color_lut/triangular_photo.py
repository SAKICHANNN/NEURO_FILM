"""Private learned triangular colour operator with positive interior Jacobian."""

import torch
from torch import nn


class TriangularPhoto(nn.Module):
    """Learn colours without allowing an own-channel slope reversal.

    Bias, nine smooth basis coefficients and two possible lower-channel terms
    per channel are bounded through tanh. Unused upper terms are masked.
    This preserves geometry, not automatically photographic appeal.
    """

    def __init__(self):
        super().__init__()
        self.parameters_raw = nn.Parameter(torch.zeros(3, 12))
        self.register_buffer("centres", torch.linspace(-4, 4, 9))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("expected BCHW RGB")
        if not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("expected finite unit RGB")
        rows = []
        p = self.parameters_raw.tanh()
        for c in range(3):
            xc = x[:, c : c + 1]
            endpoint = (xc == 0) | (xc == 1)
            z = torch.logit(torch.where(endpoint, torch.full_like(xc, 0.5), xc))
            delta = 2 * p[c, 0]
            for k in range(9):
                delta = delta + (0.5 / 9) * p[c, k + 1] * torch.tanh(
                    z - self.centres[k]
                )
            for j in range(c):
                delta = delta + 0.5 * p[c, 10 + j] * (x[:, j : j + 1] - 0.5)
            y = torch.sigmoid(z + delta)
            rows.append(torch.where(endpoint, xc, y))
        return torch.cat(rows, dim=1)
