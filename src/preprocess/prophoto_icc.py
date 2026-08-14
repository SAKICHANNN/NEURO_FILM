"""Strict high-precision decoding for matrix-shaper ProPhoto RGB ICC profiles."""

from __future__ import annotations

import struct

import numpy as np


class ProPhotoICCError(ValueError):
    """Raised when an ICC profile is not the supported ProPhoto subset."""


_D50 = np.asarray([0.9642, 1.0, 0.8249], dtype=np.float64)
_PROPHOTO_TO_XYZ_D50 = np.asarray(
    [
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0, 0.0, 0.82521],
    ],
    dtype=np.float64,
)
_D50_TO_D65_BRADFORD = np.asarray(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float64,
)
_XYZ_D65_TO_LINEAR_REC2020 = np.asarray(
    [
        [30757411 / 17917100, -6372589 / 17917100, -4539589 / 17917100],
        [-19765991 / 29648200, 47925759 / 29648200, 467509 / 29648200],
        [792561 / 44930125, -1921689 / 44930125, 42328811 / 44930125],
    ],
    dtype=np.float64,
)


def _tag_table(profile: bytes) -> dict[bytes, bytes]:
    if (
        len(profile) < 132
        or profile[16:20] != b"RGB "
        or profile[20:24] != b"XYZ "
        or profile[36:40] != b"acsp"
    ):
        raise ProPhotoICCError("ICC profile is not an RGB/XYZ profile")
    declared_size = struct.unpack_from(">I", profile, 0)[0]
    count = struct.unpack_from(">I", profile, 128)[0]
    if declared_size > len(profile) or count > 256 or 132 + count * 12 > len(profile):
        raise ProPhotoICCError("ICC tag table is invalid")
    tags: dict[bytes, bytes] = {}
    for index in range(count):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if signature in tags or size < 8 or offset + size > declared_size:
            raise ProPhotoICCError("ICC tag entry is invalid")
        tags[signature] = profile[offset : offset + size]
    return tags


def _xyz_tag(tags: dict[bytes, bytes], signature: bytes) -> np.ndarray:
    payload = tags.get(signature, b"")
    if len(payload) < 20 or payload[:4] != b"XYZ ":
        raise ProPhotoICCError(f"ICC profile lacks a valid {signature!r} tag")
    values = np.asarray(
        struct.unpack_from(">iii", payload, 8), dtype=np.float64
    )
    return values / 65536.0


def _gamma_tag(tags: dict[bytes, bytes], signature: bytes) -> float:
    payload = tags.get(signature, b"")
    if len(payload) >= 16 and payload[:4] == b"para":
        function_type = struct.unpack_from(">H", payload, 8)[0]
        if function_type != 0:
            raise ProPhotoICCError(
                "only type-0 ICC parametricCurveType TRCs are supported"
            )
        return struct.unpack_from(">i", payload, 12)[0] / 65536.0
    if len(payload) < 14 or payload[:4] != b"curv":
        raise ProPhotoICCError(f"ICC profile lacks a valid {signature!r} tag")
    count = struct.unpack_from(">I", payload, 8)[0]
    if count != 1:
        raise ProPhotoICCError("only one-parameter ICC curveType TRCs are supported")
    return struct.unpack_from(">H", payload, 12)[0] / 256.0


def prophoto_matrix_shaper_facts(profile: bytes) -> dict[str, object]:
    """Validate and return the exact transform encoded by a ProPhoto ICC profile."""

    if not isinstance(profile, bytes):
        raise TypeError("profile must be bytes")
    tags = _tag_table(profile)
    matrix = np.column_stack(
        [_xyz_tag(tags, signature) for signature in (b"rXYZ", b"gXYZ", b"bXYZ")]
    )
    white = _xyz_tag(tags, b"wtpt")
    gammas = np.asarray(
        [_gamma_tag(tags, signature) for signature in (b"rTRC", b"gTRC", b"bTRC")],
        dtype=np.float64,
    )
    matrix_error = float(np.max(np.abs(matrix - _PROPHOTO_TO_XYZ_D50)))
    white_error = float(np.max(np.abs(white - _D50)))
    if matrix_error > 5e-4 or white_error > 5e-4:
        raise ProPhotoICCError("ICC matrix/white point is not the ProPhoto RGB family")
    if float(np.max(np.abs(gammas - gammas[0]))) > 0.0 or not 1.79 <= gammas[0] <= 1.81:
        raise ProPhotoICCError("ICC TRCs are not the supported shared ProPhoto gamma")
    return {
        "gamma": float(gammas[0]),
        "matrix": matrix,
        "white_point": white,
        "maximum_prophoto_matrix_absolute_error": matrix_error,
        "maximum_d50_white_absolute_error": white_error,
    }


def decode_prophoto_rgb16_to_linear_rec2020(
    encoded: np.ndarray, profile: bytes
) -> np.ndarray:
    """Decode uint16 ProPhoto RGB to unclipped float32 linear Rec.2020."""

    if not isinstance(encoded, np.ndarray):
        raise TypeError("encoded must be a numpy ndarray")
    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError("encoded must be HxWx3 uint16 RGB")
    facts = prophoto_matrix_shaper_facts(profile)
    normalized = encoded.astype(np.float64) / 65535.0
    linear_prophoto = np.power(normalized, float(facts["gamma"]))
    xyz_d50 = linear_prophoto @ np.asarray(facts["matrix"], dtype=np.float64).T
    xyz_d65 = xyz_d50 @ _D50_TO_D65_BRADFORD.T
    rec2020 = xyz_d65 @ _XYZ_D65_TO_LINEAR_REC2020.T
    if not np.isfinite(rec2020).all():
        raise ProPhotoICCError("ICC conversion produced non-finite pixels")
    return np.asarray(rec2020, dtype=np.float32)


__all__ = [
    "ProPhotoICCError",
    "decode_prophoto_rgb16_to_linear_rec2020",
    "prophoto_matrix_shaper_facts",
]
