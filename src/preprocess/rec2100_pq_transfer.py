"""Private absolute Rec.2020 to full-range BT.2100 PQ RGB16 publication."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .png_stream import StreamingRec2100PqPngWriter

PQ_MAXIMUM_NITS = 10000.0
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = (2523.0 / 4096.0) * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = (2413.0 / 4096.0) * 32.0
PQ_C3 = (2392.0 / 4096.0) * 32.0
PQ_RGB16_MAXIMUM = 65535


def _absolute_array(values: np.ndarray) -> np.ndarray:
    absolute = np.asarray(values, dtype=np.float64)
    if absolute.ndim < 1 or absolute.shape[-1] != 3:
        raise ValueError("absolute Rec.2020 input must end in three RGB channels")
    if not np.all(np.isfinite(absolute)):
        raise ValueError("absolute Rec.2020 input must be finite")
    if np.any(absolute < 0.0) or np.any(absolute > PQ_MAXIMUM_NITS):
        raise ValueError("absolute Rec.2020 input must be inside [0, 10000] cd/m2")
    return absolute


def absolute_rec2020_cdm2_to_pq(values: np.ndarray) -> np.ndarray:
    """Apply the BT.2100 reference PQ inverse EOTF component-wise."""

    absolute = _absolute_array(values)
    normalized = absolute / PQ_MAXIMUM_NITS
    powered = np.power(normalized, PQ_M1)
    encoded = np.power(
        (PQ_C1 + PQ_C2 * powered) / (1.0 + PQ_C3 * powered),
        PQ_M2,
    )
    if not np.all(np.isfinite(encoded)) or np.any(encoded < 0.0) or np.any(encoded > 1.0):
        raise RuntimeError("BT.2100 PQ encoding escaped full range")
    return np.asarray(encoded, dtype=np.float64)


def pq_to_absolute_rec2020_cdm2(values: np.ndarray) -> np.ndarray:
    """Apply the BT.2100 reference PQ EOTF component-wise."""

    encoded = np.asarray(values, dtype=np.float64)
    if encoded.ndim < 1 or encoded.shape[-1] != 3:
        raise ValueError("PQ input must end in three RGB channels")
    if not np.all(np.isfinite(encoded)):
        raise ValueError("PQ input must be finite")
    if np.any(encoded < 0.0) or np.any(encoded > 1.0):
        raise ValueError("PQ input must be inside [0, 1]")
    powered = np.power(encoded, 1.0 / PQ_M2)
    numerator = np.maximum(powered - PQ_C1, 0.0)
    denominator = PQ_C2 - PQ_C3 * powered
    if np.any(denominator <= 0.0):
        raise RuntimeError("BT.2100 PQ EOTF denominator is not positive")
    absolute = PQ_MAXIMUM_NITS * np.power(numerator / denominator, 1.0 / PQ_M1)
    if not np.all(np.isfinite(absolute)):
        raise RuntimeError("BT.2100 PQ decoding produced nonfinite values")
    return np.asarray(absolute, dtype=np.float64)


def absolute_rec2020_cdm2_to_pq_rgb16(values: np.ndarray) -> np.ndarray:
    """Encode absolute Rec.2020 and quantize full-range RGB16 half-up."""

    encoded = absolute_rec2020_cdm2_to_pq(values)
    samples = np.floor(encoded * PQ_RGB16_MAXIMUM + 0.5).astype(np.uint16)
    return np.ascontiguousarray(samples)


def save_absolute_rec2020_cdm2_to_pq_rgb16_png(
    values: np.ndarray,
    path: Path,
    *,
    row_count: int = 64,
) -> tuple[str, np.ndarray]:
    """Create one deterministic PQ PNG and return its file hash and samples."""

    output = Path(path)
    if output.suffix.casefold() != ".png":
        raise ValueError("PQ output must use a .png extension")
    if output.exists():
        raise FileExistsError("PQ output must be create-only")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("row_count must be a positive integer")
    absolute = _absolute_array(values)
    if absolute.ndim != 3:
        raise ValueError("PQ PNG input must be HxWx3")
    samples = absolute_rec2020_cdm2_to_pq_rgb16(absolute)
    height, width, _ = samples.shape
    writer = StreamingRec2100PqPngWriter(
        output,
        width=width,
        height=height,
        compression_level=0,
    )
    try:
        for start in range(0, height, row_count):
            writer.write_rows(start, samples[start : start + row_count])
        file_sha256 = writer.finish()
    except BaseException:
        writer.abort()
        raise
    return file_sha256, samples


__all__ = [
    "PQ_C1",
    "PQ_C2",
    "PQ_C3",
    "PQ_M1",
    "PQ_M2",
    "PQ_MAXIMUM_NITS",
    "PQ_RGB16_MAXIMUM",
    "absolute_rec2020_cdm2_to_pq",
    "absolute_rec2020_cdm2_to_pq_rgb16",
    "pq_to_absolute_rec2020_cdm2",
    "save_absolute_rec2020_cdm2_to_pq_rgb16_png",
]
