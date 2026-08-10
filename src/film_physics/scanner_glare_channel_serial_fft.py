"""Channel-serial block-FFT scanner-glare execution."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

from .scanner_glare import ScannerGlareDomainError
from .scanner_glare_block_fft import _symmetric_indices


def apply_scanner_glare_channel_serial_block_fft(
    transmittance: np.ndarray,
    kernel: np.ndarray,
    *,
    flare_fraction: float,
    row_chunk: int,
) -> np.ndarray:
    """Apply block FFT while retaining only one channel halo and pad at a time."""
    values = np.asarray(transmittance, dtype=np.float64)
    spread = np.asarray(kernel, dtype=np.float64)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] != 3):
        raise ScannerGlareDomainError("transmittance must be HxW or HxWx3")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ScannerGlareDomainError("transmittance must be finite in [0, 1]")
    if (
        spread.ndim != 2
        or spread.shape[0] != spread.shape[1]
        or spread.shape[0] % 2 == 0
        or not np.all(np.isfinite(spread))
        or np.any(spread < 0.0)
        or not math.isclose(float(np.sum(spread)), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ScannerGlareDomainError("kernel must be positive normalized odd square")
    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk < 1:
        raise ScannerGlareDomainError("row chunk must be a positive integer")
    if not math.isfinite(flare_fraction) or not 0.0 <= flare_fraction < 1.0:
        raise ScannerGlareDomainError("flare fraction must be in [0, 1)")

    height = values.shape[0]
    radius = spread.shape[0] // 2
    planes = values[..., None] if values.ndim == 2 else values
    output = (1.0 - flare_fraction) * planes.copy()
    for start in range(0, height, row_chunk):
        stop = min(start + row_chunk, height)
        raw_rows = np.arange(start - radius, stop + radius, dtype=np.int64)
        mapped_rows = _symmetric_indices(raw_rows, height)
        for channel in range(planes.shape[-1]):
            strip = planes[mapped_rows, :, channel]
            padded = np.pad(strip, ((0, 0), (radius, radius)), mode="symmetric")
            blurred = fftconvolve(padded, spread, mode="valid")
            output[start:stop, :, channel] += flare_fraction * blurred
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1e-12)
        or np.any(output > 1.0 + 1e-12)
    ):
        raise RuntimeError("channel-serial block FFT left bounded transmittance domain")
    bounded = np.clip(output, 0.0, 1.0)
    return bounded[..., 0] if values.ndim == 2 else bounded
