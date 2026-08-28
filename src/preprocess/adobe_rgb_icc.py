"""Strict RGB16 decoding for matrix-shaper profiles compatible with Adobe RGB (1998)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .prophoto_icc import (
    ProPhotoICCError,
    _gamma_tag,
    _tag_table,
    _xyz_tag,
    d50_xyz_to_linear_rec2020,
)


class AdobeRGBICCError(ValueError):
    """Raised when an ICC profile is outside the supported strict subset."""


_D50 = np.asarray([0.9642, 1.0, 0.8249], dtype=np.float64)
_D65 = np.asarray([0.9504559, 1.0, 1.0890578], dtype=np.float64)
_ADOBE_RGB_TO_XYZ_D50 = np.asarray(
    [
        [0.6097559, 0.2052401, 0.1492240],
        [0.3111242, 0.6256560, 0.0632197],
        [0.0194811, 0.0608902, 0.7448387],
    ],
    dtype=np.float64,
)
_GAMMA = 563.0 / 256.0


def adobe_rgb_icc_facts(profile: bytes) -> dict[str, object]:
    """Validate a strict matrix-shaper profile compatible with Adobe RGB (1998)."""

    if not isinstance(profile, bytes):
        raise TypeError("profile must be bytes")
    try:
        tags = _tag_table(profile)
        matrix = np.column_stack(
            [_xyz_tag(tags, signature) for signature in (b"rXYZ", b"gXYZ", b"bXYZ")]
        )
        white = _xyz_tag(tags, b"wtpt")
        gammas = np.asarray(
            [_gamma_tag(tags, signature) for signature in (b"rTRC", b"gTRC", b"bTRC")],
            dtype=np.float64,
        )
    except ProPhotoICCError as exc:
        raise AdobeRGBICCError(str(exc)) from exc

    matrix_error = float(np.max(np.abs(matrix - _ADOBE_RGB_TO_XYZ_D50)))
    d50_error = float(np.max(np.abs(white - _D50)))
    d65_error = float(np.max(np.abs(white - _D65)))
    gamma_error = float(np.max(np.abs(gammas - _GAMMA)))
    if matrix_error > 5e-4:
        raise AdobeRGBICCError("ICC colourants are not the supported Adobe RGB family")
    if min(d50_error, d65_error) > 5e-4:
        raise AdobeRGBICCError("ICC media white is neither supported D50 nor D65")
    if float(np.max(np.abs(gammas - gammas[0]))) > 0.0 or gamma_error > 2e-5:
        raise AdobeRGBICCError("ICC TRCs are not the supported shared Adobe RGB gamma")
    return {
        "transform_kind": "matrix-shaper-shared-gamma",
        "gamma": float(gammas[0]),
        "matrix": matrix,
        "white_point": white,
        "maximum_matrix_absolute_error": matrix_error,
        "maximum_gamma_absolute_error": gamma_error,
        "maximum_supported_white_absolute_error": min(d50_error, d65_error),
    }


@lru_cache(maxsize=8)
def _prepared_transfer(profile: bytes) -> tuple[np.ndarray, np.ndarray]:
    facts = adobe_rgb_icc_facts(profile)
    normalized = np.arange(65536, dtype=np.float64) / 65535.0
    transfer = np.power(normalized, float(facts["gamma"]))
    matrix = np.asarray(facts["matrix"], dtype=np.float64).copy()
    transfer.flags.writeable = False
    matrix.flags.writeable = False
    return transfer, matrix


def decode_adobe_rgb16_to_linear_rec2020(
    encoded: np.ndarray, profile: bytes
) -> np.ndarray:
    """Decode strict compatible RGB16 samples to unclipped linear Rec.2020."""

    if not isinstance(encoded, np.ndarray):
        raise TypeError("encoded must be a numpy ndarray")
    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError("encoded must be HxWx3 uint16 RGB")
    transfer, matrix = _prepared_transfer(profile)
    xyz_d50 = transfer[encoded] @ matrix.T
    return np.asarray(d50_xyz_to_linear_rec2020(xyz_d50), dtype=np.float32)


__all__ = [
    "AdobeRGBICCError",
    "adobe_rgb_icc_facts",
    "decode_adobe_rgb16_to_linear_rec2020",
]
