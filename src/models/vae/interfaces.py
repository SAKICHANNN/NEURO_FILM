import torch
import torch.nn as nn


class HnDToneMapper(nn.Module):
    """H&D characteristic curve tone mapping via per-channel 1D LUT.

    Operates on linear RGB input. H&D curves are defined in
    log10(exposure) → density space and converted to RGB LUTs on init.
    """

    def __init__(self, curve_csv: str, n_points: int = 256):
        """
        Args:
            curve_csv: Path to digitized H&D curve CSV
            n_points: Number of LUT sampling points
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) float32 linear RGB [0,1] → (B,3,H,W) tone-mapped."""
        raise NotImplementedError
