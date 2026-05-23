import torch
import torch.nn as nn
from typing import Optional
from PIL import Image


class PostProcessGrain:
    """Optional film grain post-processing (filmgrader or Newson)."""

    def __init__(self, rms_granularity: float = 10.0, method: str = "filmgrainer"):
        raise NotImplementedError

    def apply(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError


class PostProcessHalation:
    """Optional halation post-processing (Gaussian scatter)."""

    def __init__(self, r_sigma: float = 0.4, g_sigma: float = 0.2,
                 b_sigma: float = 0.1, threshold: float = 0.85):
        raise NotImplementedError

    def apply(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError
