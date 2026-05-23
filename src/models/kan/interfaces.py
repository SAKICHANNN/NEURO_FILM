import torch
import torch.nn as nn


class HalationSimulator(nn.Module):
    """Multi-channel Gaussian scatter to simulate film halation.

    Red channel gets largest scatter radius (bounces back through
    anti-halation layer), green medium, blue smallest.

    Consumer: pipeline_full.py
    """

    def __init__(self, r_sigma: float = 0.6, g_sigma: float = 0.3, b_sigma: float = 0.15, threshold: float = 0.85):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B,3,H,W)"""
        raise NotImplementedError


class FilmGrainRenderer(nn.Module):
    """Film grain rendering — wraps filmgrainer (MIT) or custom Newson implementation.

    Consumer: pipeline_full.py
    """

    def __init__(self, grain_radius: float = 0.1, grain_sigma: float = 0.0, n_monte_carlo: int = 400):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B,3,H,W)"""
        raise NotImplementedError
