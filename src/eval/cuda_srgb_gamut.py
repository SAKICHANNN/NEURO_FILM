"""Isolated CUDA implementation of the legacy sRGB Lab gamut compressor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CudaSRGBGamutError(RuntimeError):
    """Raised when the isolated CUDA gamut contract is not satisfied."""


@dataclass(frozen=True)
class CudaGamutExecution:
    output_lab: np.ndarray
    peak_allocated_bytes: int


def _validated_pair(
    source_lab: np.ndarray, target_lab: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(source_lab, np.ndarray) or not isinstance(target_lab, np.ndarray):
        raise TypeError("source_lab and target_lab must be numpy arrays")
    if (
        source_lab.dtype != np.float32
        or target_lab.dtype != np.float32
        or source_lab.ndim != 3
        or source_lab.shape[-1] != 3
        or target_lab.shape != source_lab.shape
        or source_lab.shape[0] == 0
        or source_lab.shape[1] == 0
    ):
        raise ValueError("source_lab and target_lab must be non-empty float32 HxWx3")
    if not np.isfinite(source_lab).all() or not np.isfinite(target_lab).all():
        raise ValueError("source_lab and target_lab must be finite")
    return np.ascontiguousarray(source_lab), np.ascontiguousarray(target_lab)


def _linear_srgb_torch(lab, torch):
    value = lab.to(dtype=torch.float32)
    y = (value[..., 0] + 16.0) / 116.0
    x = y + value[..., 1] / 500.0
    z = y - value[..., 2] / 200.0
    delta = 6.0 / 29.0

    def inv_f(component):
        return torch.where(
            component > delta,
            component * component * component,
            (3.0 * delta * delta) * (component - 4.0 / 29.0),
        )

    xyz = torch.stack((0.95047 * inv_f(x), inv_f(y), 1.08883 * inv_f(z)), dim=-1)
    matrix = torch.tensor(
        (
            (3.2404542, -1.5371385, -0.4985314),
            (-0.9692660, 1.8760108, 0.0415560),
            (0.0556434, -0.2040259, 1.0572252),
        ),
        dtype=torch.float32,
        device=lab.device,
    )
    return torch.matmul(xyz, matrix.transpose(0, 1))


def compress_to_srgb_gamut_cuda(
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    *,
    iterations: int = 14,
    device_index: int = 0,
) -> CudaGamutExecution:
    """Apply the frozen legacy segment compressor on one CUDA device."""

    source, target = _validated_pair(source_lab, target_lab)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CudaSRGBGamutError("PyTorch is unavailable") from exc
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise CudaSRGBGamutError("CUDA is unavailable")
    if device_index >= torch.cuda.device_count():
        raise CudaSRGBGamutError("CUDA device_index is unavailable")

    device = torch.device(f"cuda:{device_index}")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        source_cuda = torch.from_numpy(source).to(device=device)
        target_cuda = torch.from_numpy(target).to(device=device)
        delta = target_cuda - source_cuda
        low = torch.zeros(source_cuda.shape[:-1] + (1,), dtype=torch.float32, device=device)
        high = torch.ones_like(low)
        for _ in range(iterations):
            mid = (low + high) * 0.5
            candidate = source_cuda + delta * mid
            linear = _linear_srgb_torch(candidate, torch)
            valid = torch.all((linear >= 0.0) & (linear <= 1.0), dim=-1, keepdim=True)
            low = torch.where(valid, mid, low)
            high = torch.where(valid, high, mid)
        result = source_cuda + delta * low
        if not bool(torch.isfinite(result).all().item()):
            raise CudaSRGBGamutError("CUDA gamut output is non-finite")
        output = result.cpu().numpy()
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    except RuntimeError as exc:
        raise CudaSRGBGamutError("CUDA gamut execution failed") from exc
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return CudaGamutExecution(
        output_lab=np.ascontiguousarray(output, dtype=np.float32),
        peak_allocated_bytes=peak,
    )


__all__ = [
    "CudaGamutExecution",
    "CudaSRGBGamutError",
    "compress_to_srgb_gamut_cuda",
]
