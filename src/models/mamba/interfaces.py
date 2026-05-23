"""Mamba backbone interface consumed by CFM velocity_net.
契约: CFM 的 VelocityFieldPredictor 调用此骨干。张量形状不变量见签名。
"""

import torch
import torch.nn as nn


class MambaBackboneInterface(nn.Module):
    """
    Contract: CFM velocity_net calls this backbone to process latent features.

    Input:  (B, 4, H/8, W/8) — VAE-encoded latent
    Output: (B, 4, H/8, W/8) — processed features, same spatial dims
    """

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, H//8, W//8) latent features from VAE encoder
            t_emb: (B, embed_dim) sinusoidal time embedding [0,1]
        Returns:
            features: (B, 4, H//8, W//8) velocity-field-conditioned features
        """
        raise NotImplementedError


class MambaSSMBlockInterface(nn.Module):
    """单层 SSM block 的接口。由 Mamba 骨干内部使用。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) — sequence format
        Returns:
            (B, L, D) — same shape
        """
        raise NotImplementedError
