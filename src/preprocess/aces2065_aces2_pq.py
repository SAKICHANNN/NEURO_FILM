"""Strict private ACES2065-1 OpenEXR to canonical ACES 2 PQ composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .aces2_canonical_pq_png import (
    publish_working_image_aces2_canonical_hdr_pq_png_v1,
)
from .aces2065_openexr import load_aces2065_openexr_working_image


def publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(
    source: Path | str,
    output: Path | str,
    *,
    row_count: int = 64,
    reverse_partition: bool = False,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Load strict AP0/D60 EXR and publish official Rec.2020-PQ RGB16 PNG."""

    working = load_aces2065_openexr_working_image(source)
    return publish_working_image_aces2_canonical_hdr_pq_png_v1(
        working,
        Path(output),
        row_count=row_count,
        reverse_partition=reverse_partition,
    )


__all__ = ["publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1"]
