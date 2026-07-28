"""Compile the frozen AO6 t15/c35 residual for its native float32 ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
from typing import Any

import numpy as np

from src.roll2film.positive_film import PositiveFilmResponseOperator


class NativeAo6ResidualProfileF32V1(ctypes.Structure):
    _MatrixRow = ctypes.c_double * 3
    _Matrix = _MatrixRow * 3
    _Vector = ctypes.c_double * 3
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("capture_matrix", _Matrix),
        ("response_midpoints", _Vector),
        ("response_slopes", _Vector),
        ("maximum_responses", _Vector),
        ("scan_matrix", _Matrix),
        ("black_endpoint", _Vector),
        ("white_endpoint", _Vector),
        ("luma_weights", _Vector),
        ("exposure_floor", ctypes.c_double),
        ("matrix_minimum_determinant", ctypes.c_double),
        ("minimum_endpoint_span", ctypes.c_double),
        ("tone_strength", ctypes.c_double),
        ("chroma_strength", ctypes.c_double),
        ("hard_low_linear", ctypes.c_double),
        ("hard_high_linear", ctypes.c_double),
        ("guard_low_linear", ctypes.c_double),
        ("guard_high_linear", ctypes.c_double),
    ]


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _encoded_srgb_to_linear_scalar(value: float) -> float:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("encoded sRGB threshold is invalid")
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def native_ao6_residual_profile_struct(
    display_payload: dict[str, Any],
) -> NativeAo6ResidualProfileF32V1:
    residual = display_payload["residual"]
    if (
        float(residual["tone_strength"]) != 0.15
        or float(residual["chroma_strength"]) != 0.35
        or residual["hard_clip_allowed"]
    ):
        raise ValueError("native AO6 residual contract drift")
    operator = PositiveFilmResponseOperator.from_dict(
        residual["operator"]
    )
    weights = np.asarray(residual["luma_weights"], dtype=np.float64)
    if (
        weights.shape != (3,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or abs(float(np.sum(weights)) - 1.0) > 1e-12
    ):
        raise ValueError("native AO6 luma weights are invalid")
    hard = float(
        residual["hard_boundary_epsilon_encoded_srgb"]
    )
    guard = float(
        residual["guard_boundary_epsilon_encoded_srgb"]
    )
    if not 0.0 <= hard < guard < 0.5:
        raise ValueError("native AO6 boundary thresholds are invalid")
    black, white = operator._raw_endpoints()
    profile = NativeAo6ResidualProfileF32V1()
    profile.struct_size = ctypes.sizeof(NativeAo6ResidualProfileF32V1)
    profile.abi_version = 1
    profile.source_component_sha256 = _payload_sha256(
        display_payload
    ).encode("ascii")
    for row in range(3):
        for column in range(3):
            profile.capture_matrix[row][column] = float(
                operator.capture_matrix[row, column]
            )
            profile.scan_matrix[row][column] = float(
                operator.scan_matrix[row, column]
            )
        profile.response_midpoints[row] = float(
            operator.response_midpoints[row]
        )
        profile.response_slopes[row] = float(
            operator.response_slopes[row]
        )
        profile.maximum_responses[row] = float(
            operator.maximum_responses[row]
        )
        profile.black_endpoint[row] = float(black[row])
        profile.white_endpoint[row] = float(white[row])
        profile.luma_weights[row] = float(weights[row])
    profile.exposure_floor = float(operator.exposure_floor)
    profile.matrix_minimum_determinant = float(
        operator.matrix_minimum_determinant
    )
    profile.minimum_endpoint_span = float(
        operator.minimum_endpoint_span
    )
    profile.tone_strength = 0.15
    profile.chroma_strength = 0.35
    profile.hard_low_linear = _encoded_srgb_to_linear_scalar(hard)
    profile.hard_high_linear = _encoded_srgb_to_linear_scalar(
        1.0 - hard
    )
    profile.guard_low_linear = _encoded_srgb_to_linear_scalar(guard)
    profile.guard_high_linear = _encoded_srgb_to_linear_scalar(
        1.0 - guard
    )
    return profile


def native_ao6_display_payload_sha256(
    display_payload: dict[str, Any],
) -> str:
    native_ao6_residual_profile_struct(display_payload)
    return _payload_sha256(display_payload)
