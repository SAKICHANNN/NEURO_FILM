"""Private exact row kernels for bounded reference-match colour conversion."""

from __future__ import annotations

import numpy as np

from src.color_engine import (
    in_working_gamut,
    lab_to_linear_rgb,
    linear_rgb_to_lab,
)


REFERENCE_MATCH_ROW_CHUNK = 128


def linear_rgb_to_lab_rows(
    pixels: np.ndarray,
    *,
    working_space: str,
) -> np.ndarray:
    """Run the authoritative pointwise conversion with bounded temporaries."""

    output = np.empty_like(pixels, dtype=np.float32)
    for y0 in range(0, pixels.shape[0], REFERENCE_MATCH_ROW_CHUNK):
        y1 = min(y0 + REFERENCE_MATCH_ROW_CHUNK, pixels.shape[0])
        output[y0:y1] = linear_rgb_to_lab(
            pixels[y0:y1],
            working_space=working_space,
        )
    return output


def lab_to_linear_rgb_rows(
    lab: np.ndarray,
    *,
    working_space: str,
) -> np.ndarray:
    """Run the authoritative pointwise inverse with bounded temporaries."""

    output = np.empty_like(lab, dtype=np.float32)
    for y0 in range(0, lab.shape[0], REFERENCE_MATCH_ROW_CHUNK):
        y1 = min(y0 + REFERENCE_MATCH_ROW_CHUNK, lab.shape[0])
        output[y0:y1] = lab_to_linear_rgb(
            lab[y0:y1],
            working_space=working_space,
        )
    return output


def in_working_gamut_rows(
    lab: np.ndarray,
    *,
    working_space: str,
    tolerance: float,
) -> bool:
    """Fail fast while bounding the authoritative gamut predicate."""

    for y0 in range(0, lab.shape[0], REFERENCE_MATCH_ROW_CHUNK):
        y1 = min(y0 + REFERENCE_MATCH_ROW_CHUNK, lab.shape[0])
        if not in_working_gamut(
            lab[y0:y1],
            working_space=working_space,
            tolerance=tolerance,
        ).all():
            return False
    return True


__all__ = [
    "REFERENCE_MATCH_ROW_CHUNK",
    "in_working_gamut_rows",
    "lab_to_linear_rgb_rows",
    "linear_rgb_to_lab_rows",
]
