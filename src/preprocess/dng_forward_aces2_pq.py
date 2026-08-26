"""Private source-bound DNG ForwardMatrix to official ACES 2 P3-PQ callable."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dng_forward_raster import (
    DngForwardRasterError,
    load_dng_forward_working_image,
)
from .ocio_aces2_output import (
    OcioAces2RuntimeError,
    apply_working_image_aces2_output,
)

TARGET = "hdr_p3d65_1000nit_rec2100_pq"


class DngForwardAces2PqError(RuntimeError):
    """Raised when the private source-bound callable cannot return valid PQ."""


def render_dng_forward_to_aces2_p3_pq(
    path: str | Path,
    *,
    expected_source_bytes: int,
    expected_source_sha256: str,
) -> np.ndarray:
    """Return owned float32 normalized PQ for one explicitly bound DNG source."""

    if (
        not isinstance(expected_source_bytes, int)
        or isinstance(expected_source_bytes, bool)
        or expected_source_bytes <= 0
    ):
        raise DngForwardAces2PqError(
            "expected_source_bytes must be a positive integer"
        )
    if (
        not isinstance(expected_source_sha256, str)
        or len(expected_source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_source_sha256)
    ):
        raise DngForwardAces2PqError(
            "expected_source_sha256 must be lowercase hexadecimal SHA-256"
        )
    try:
        working = load_dng_forward_working_image(
            path,
            expected_source_bytes=expected_source_bytes,
            expected_source_sha256=expected_source_sha256,
        )
        rendered = apply_working_image_aces2_output(working, TARGET)
    except (DngForwardRasterError, OcioAces2RuntimeError) as exc:
        raise DngForwardAces2PqError(str(exc)) from exc

    output = np.array(rendered, dtype=np.float32, order="C", copy=True)
    if output.ndim != 3 or output.shape[2] != 3 or output.size == 0:
        raise DngForwardAces2PqError("official ACES 2 output must be non-empty HxWx3")
    if not np.isfinite(output).all():
        raise DngForwardAces2PqError("official ACES 2 output is non-finite")
    if np.any((output < 0.0) | (output > 1.0)):
        raise DngForwardAces2PqError("official ACES 2 output is outside normalized PQ")
    return output


__all__ = [
    "TARGET",
    "DngForwardAces2PqError",
    "render_dng_forward_to_aces2_p3_pq",
]
