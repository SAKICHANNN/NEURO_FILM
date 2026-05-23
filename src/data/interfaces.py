from typing import Optional
import torch
from torch.utils.data import Dataset


class UnpairedFilmDataset(Dataset):
    """Unpaired film style dataset for CUT/CycleGAN training.

    Domain A: digital images (FiveK inputs, COCO, etc.)
    Domain B: film domain images (FilmSet, web-scraped film scans, etc.)
    """

    def __init__(self, data_dir: str, domain: str = "digital", resolution: int = 256):
        """
        Args:
            data_dir: Root path (contains digital/ and film/ subdirs)
            domain: 'digital' or 'film'
            resolution: Square crop size
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns (3, H, W) float32, linear RGB, [0,1]"""
        raise NotImplementedError


class RAWPipelineInterface:
    """RAW file → linear RGB processing pipeline."""

    def process(self, raw_path: str) -> tuple[torch.Tensor, dict]:
        """Returns (3,H,W) float32 linear RGB [0,1] and metadata dict."""
        raise NotImplementedError


class HnDCurveLoader:
    """Loads digitized H&D curves from CSV → builds 1D LUTs."""

    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: Path to CSV with columns [log_exposure, density_r, density_g, density_b]
        """
        raise NotImplementedError

    def build_lut(self, n_points: int = 256) -> torch.Tensor:
        """Returns (3, n_points) float32 1D LUT for per-channel tone mapping."""
        raise NotImplementedError


class HalationSimulator:
    """Multi-channel Gaussian halation scatter."""

    def __init__(self, r_sigma: float = 0.6, g_sigma: float = 0.3, b_sigma: float = 0.15, threshold: float = 0.85):
        raise NotImplementedError

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B,3,H,W)"""
        raise NotImplementedError


class FilmGrainRenderer:
    """Film grain application via filmgrainer or custom Newson implementation."""

    def __init__(self, method: str = "filmgrainer", grain_params: Optional[dict] = None):
        """
        Args:
            method: 'filmgrainer' (MIT licensed) or 'newson' (self-implemented)
            grain_params: Film-specific grain parameters (RMS granularity, etc.)
        """
        raise NotImplementedError

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B,3,H,W)"""
        raise NotImplementedError
