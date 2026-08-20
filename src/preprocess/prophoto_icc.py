"""Strict high-precision decoding for supported ProPhoto RGB ICC profiles."""

from __future__ import annotations

import struct
from functools import lru_cache

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
_ICC_PCS_XYZ_SCALE = 65535.0 / 32768.0
_ROMM_REFERENCE_MEDIA_WHITE_SCALE = 0.89


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


def _aligned(value: int) -> int:
    return (value + 3) & ~3


def _embedded_curve(
    payload: bytes, offset: int
) -> tuple[dict[str, object], int]:
    if offset < 0 or offset + 12 > len(payload):
        raise ProPhotoICCError("ICC embedded curve offset is invalid")
    signature = payload[offset : offset + 4]
    if signature == b"curv":
        count = struct.unpack_from(">I", payload, offset + 8)[0]
        size = 12 + count * 2
        if offset + size > len(payload):
            raise ProPhotoICCError("ICC embedded curve is truncated")
        if count != 0:
            raise ProPhotoICCError("only identity embedded curveType is supported")
        return {"kind": "identity"}, _aligned(offset + size)
    if signature != b"para":
        raise ProPhotoICCError("ICC embedded curve type is unsupported")
    function_type = struct.unpack_from(">H", payload, offset + 8)[0]
    parameter_counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
    count = parameter_counts.get(function_type)
    if count is None or offset + 12 + count * 4 > len(payload):
        raise ProPhotoICCError("ICC parametric curve is invalid")
    parameters = np.asarray(
        struct.unpack_from(">" + "i" * count, payload, offset + 12),
        dtype=np.float64,
    ) / 65536.0
    return {
        "kind": "parametric",
        "function_type": function_type,
        "parameters": parameters,
    }, _aligned(offset + 12 + count * 4)


def _curve_set(payload: bytes, offset: int, count: int) -> list[dict[str, object]]:
    curves: list[dict[str, object]] = []
    cursor = offset
    for _ in range(count):
        curve, cursor = _embedded_curve(payload, cursor)
        curves.append(curve)
    return curves


def _mab_prophoto_facts(tags: dict[bytes, bytes]) -> dict[str, object]:
    payload = tags.get(b"A2B0", b"")
    if len(payload) < 32 or payload[:4] != b"mAB ":
        raise ProPhotoICCError("ICC profile lacks a supported A2B0 mAB transform")
    input_channels, output_channels = struct.unpack_from(">BB", payload, 8)
    b_offset, matrix_offset, m_offset, clut_offset, a_offset = struct.unpack_from(
        ">IIIII", payload, 12
    )
    if (
        input_channels != 3
        or output_channels != 3
        or not b_offset
        or not matrix_offset
        or not m_offset
        or clut_offset != 0
        or a_offset != 0
    ):
        raise ProPhotoICCError("ICC mAB topology is outside the supported subset")
    b_curves = _curve_set(payload, b_offset, output_channels)
    m_curves = _curve_set(payload, m_offset, output_channels)
    if any(curve["kind"] != "identity" for curve in b_curves):
        raise ProPhotoICCError("ICC mAB B curves must be identity")
    if any(
        curve["kind"] != "parametric" or curve["function_type"] != 3
        for curve in m_curves
    ):
        raise ProPhotoICCError("ICC mAB M curves must be type-3 parametric curves")
    parameters = np.stack(
        [np.asarray(curve["parameters"], dtype=np.float64) for curve in m_curves]
    )
    expected = np.asarray([1.8, 1.0, 0.0, 1.0 / 16.0, 1.0 / 32.0])
    if float(np.max(np.abs(parameters - parameters[0]))) > 0.0 or float(
        np.max(np.abs(parameters[0] - expected))
    ) > 5e-5:
        raise ProPhotoICCError("ICC mAB curves are not the supported ROMM transfer")
    if matrix_offset + 48 > len(payload):
        raise ProPhotoICCError("ICC mAB matrix is truncated")
    matrix_values = np.asarray(
        struct.unpack_from(">" + "i" * 12, payload, matrix_offset),
        dtype=np.float64,
    ) / 65536.0
    matrix = matrix_values[:9].reshape(3, 3)
    offset = matrix_values[9:]
    effective_matrix = matrix * _ICC_PCS_XYZ_SCALE
    white = _xyz_tag(tags, b"wtpt")
    pcs_encoded_white = (matrix.sum(axis=1) + offset) * _ICC_PCS_XYZ_SCALE
    matrix_error = float(np.max(np.abs(effective_matrix - _PROPHOTO_TO_XYZ_D50)))
    white_error = float(
        np.max(np.abs(white - _D50 * _ROMM_REFERENCE_MEDIA_WHITE_SCALE))
    )
    pcs_white_error = float(np.max(np.abs(pcs_encoded_white - _D50)))
    if matrix_error > 2e-3 or white_error > 5e-4 or pcs_white_error > 5e-3:
        raise ProPhotoICCError("ICC mAB matrix/white point is not the ProPhoto family")
    return {
        "transform_kind": "mab-romm-type3-matrix-offset-pcsxyz",
        "transfer_kind": "parametric-type3",
        "transfer_parameters": parameters[0],
        "matrix": matrix,
        "offset": offset,
        "pcs_xyz_scale": _ICC_PCS_XYZ_SCALE,
        "white_point": white,
        "maximum_prophoto_matrix_absolute_error": matrix_error,
        "maximum_reference_media_white_absolute_error": white_error,
        "maximum_pcs_encoded_white_absolute_error": pcs_white_error,
    }


def prophoto_icc_facts(profile: bytes) -> dict[str, object]:
    """Validate and return an exact supported ProPhoto ICC transform."""

    if not isinstance(profile, bytes):
        raise TypeError("profile must be bytes")
    tags = _tag_table(profile)
    if all(signature in tags for signature in (b"rXYZ", b"gXYZ", b"bXYZ")):
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
            "transform_kind": "matrix-shaper-shared-gamma",
            "transfer_kind": "gamma",
            "gamma": float(gammas[0]),
            "matrix": matrix,
            "offset": np.zeros(3, dtype=np.float64),
            "pcs_xyz_scale": 1.0,
            "white_point": white,
            "maximum_prophoto_matrix_absolute_error": matrix_error,
            "maximum_d50_white_absolute_error": white_error,
        }
    return _mab_prophoto_facts(tags)


def prophoto_matrix_shaper_facts(profile: bytes) -> dict[str, object]:
    """Validate and return the exact transform encoded by a ProPhoto ICC profile."""

    facts = prophoto_icc_facts(profile)
    if facts["transform_kind"] != "matrix-shaper-shared-gamma":
        raise ProPhotoICCError("ICC profile is not a matrix-shaper transform")
    return facts


@lru_cache(maxsize=8)
def _prepared_rgb16_transfer(
    profile: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    facts = prophoto_icc_facts(profile)
    normalized = np.arange(65536, dtype=np.float64) / 65535.0
    if facts["transfer_kind"] == "gamma":
        transfer = np.power(normalized, float(facts["gamma"]))
    elif facts["transfer_kind"] == "parametric-type3":
        gamma, a, b, c, d = np.asarray(
            facts["transfer_parameters"], dtype=np.float64
        )
        transfer = np.where(
            normalized >= d,
            np.power(a * normalized + b, gamma),
            c * normalized,
        )
    else:  # pragma: no cover - facts are produced only by validated branches.
        raise ProPhotoICCError("ICC transfer kind is unsupported")
    matrix = np.asarray(facts["matrix"], dtype=np.float64).copy()
    offset = np.asarray(facts["offset"], dtype=np.float64).copy()
    for array in (transfer, matrix, offset):
        array.flags.writeable = False
    return transfer, matrix, offset, float(facts["pcs_xyz_scale"])


def decode_prophoto_rgb16_to_linear_rec2020(
    encoded: np.ndarray, profile: bytes
) -> np.ndarray:
    """Decode uint16 ProPhoto RGB to unclipped float32 linear Rec.2020."""

    if not isinstance(encoded, np.ndarray):
        raise TypeError("encoded must be a numpy ndarray")
    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError("encoded must be HxWx3 uint16 RGB")
    transfer, matrix, offset, pcs_xyz_scale = _prepared_rgb16_transfer(profile)
    linear_prophoto = transfer[encoded]
    xyz_d50 = (
        linear_prophoto @ matrix.T + offset
    ) * pcs_xyz_scale
    xyz_d65 = xyz_d50 @ _D50_TO_D65_BRADFORD.T
    rec2020 = xyz_d65 @ _XYZ_D65_TO_LINEAR_REC2020.T
    if not np.isfinite(rec2020).all():
        raise ProPhotoICCError("ICC conversion produced non-finite pixels")
    return np.asarray(rec2020, dtype=np.float32)


__all__ = [
    "ProPhotoICCError",
    "decode_prophoto_rgb16_to_linear_rec2020",
    "prophoto_icc_facts",
    "prophoto_matrix_shaper_facts",
]
