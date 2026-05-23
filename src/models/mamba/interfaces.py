import torch
import torch.nn as nn


class ColorTransferInterface(nn.Module):
    """Interface for domain-to-domain color style transfer.

    Consumer: pipeline_full.py
    Implementer: CUT/CycleGAN generator or 3D LUT predictor
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) digital linear RGB → (B,3,H,W) film-colored linear RGB.

        The output should preserve content structure while transforming
        color distribution to match the target film domain.
        """
        raise NotImplementedError
