"""Private canonical RGB16 PNG publisher for the P226 P3-D65 PQ target."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .canonical_pq_png import CanonicalStreamingRec2100PqPngWriter
from .ocio_aces2_output import apply_aces2_output_packed

TARGET = "hdr_p3d65_1000nit_rec2100_pq"


def publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
    acescg: np.ndarray,
    path: Path,
    *,
    row_count: int = 7,
    reverse_partition: bool = False,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Publish one exact OCIO P3-D65 PQ result through the canonical PNG rail."""

    source = np.asarray(acescg)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[2:] != (3,)
        or source.shape[0] <= 0
        or source.shape[1] <= 0
        or not source.flags.c_contiguous
        or not np.isfinite(source).all()
    ):
        raise ValueError("ACEScg input must be finite non-empty contiguous HxWx3 float32")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("row_count must be a positive integer")
    output = Path(path)
    if output.suffix.casefold() != ".png":
        raise ValueError("canonical P3-D65 PQ output requires a .png extension")
    if output.exists():
        raise FileExistsError("canonical P3-D65 PQ output is create-only")

    encoded_rows = apply_aces2_output_packed(source.reshape(-1, 3), TARGET)
    encoded = np.ascontiguousarray(encoded_rows.reshape(source.shape))
    if np.any(encoded < 0.0) or np.any(encoded > 1.0):
        raise ValueError("official P3-D65 PQ output must remain inside [0, 1]")
    samples = np.ascontiguousarray(
        np.floor(encoded.astype(np.float64) * 65535.0 + 0.5).astype(np.uint16)
    )

    height, width, _ = samples.shape
    sizes = [row_count] * (height // row_count)
    if height % row_count:
        sizes.append(height % row_count)
    if reverse_partition:
        sizes.reverse()
    writer = CanonicalStreamingRec2100PqPngWriter(output, width=width, height=height)
    try:
        start = 0
        for size in sizes:
            writer.write_rows(start, samples[start : start + size])
            start += size
        digest = writer.finish()
    except BaseException:
        writer.abort()
        raise
    return digest, samples, encoded


__all__ = ["TARGET", "publish_acescg_p3d65_1000nit_canonical_pq_png_v1"]
