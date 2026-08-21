"""Private WorkingImage to official ACES 2 HDR PQ RGB16 PNG bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ocio_aces2_output import apply_working_image_aces2_output
from .png_stream import StreamingRec2100PqPngWriter
from .types import WorkingImage


def publish_working_image_aces2_hdr_pq_png_v1(
    working: WorkingImage,
    path: Path,
    *,
    row_count: int = 64,
    reverse_partition: bool = False,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Apply the pinned ACES 2 HDR view and publish exact full-range RGB16."""

    output = Path(path)
    if output.suffix.casefold() != ".png":
        raise ValueError("ACES 2 HDR PQ output must use a .png extension")
    if output.exists():
        raise FileExistsError("ACES 2 HDR PQ output must be create-only")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("row_count must be a positive integer")
    encoded = apply_working_image_aces2_output(working, "hdr_rec2020_pq")
    if not np.isfinite(encoded).all() or np.any(encoded < 0.0) or np.any(encoded > 1.0):
        raise ValueError("ACES 2 HDR PQ output must be finite and inside [0, 1]")
    samples = np.ascontiguousarray(
        np.floor(encoded.astype(np.float64) * 65535.0 + 0.5).astype(np.uint16)
    )
    height, width, _ = samples.shape
    writer = StreamingRec2100PqPngWriter(
        output,
        width=width,
        height=height,
        compression_level=0,
    )
    try:
        sizes = [row_count] * (height // row_count)
        if height % row_count:
            sizes.append(height % row_count)
        if reverse_partition:
            sizes.reverse()
        start = 0
        for size in sizes:
            writer.write_rows(start, samples[start : start + size])
            start += size
        file_sha256 = writer.finish()
    except BaseException:
        writer.abort()
        raise
    return file_sha256, samples, encoded


__all__ = ["publish_working_image_aces2_hdr_pq_png_v1"]
