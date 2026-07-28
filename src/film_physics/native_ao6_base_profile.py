"""Compile the active AO6 source-context base for its native float32 ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
from typing import Any

import numpy as np

from src.color_engine.safe_lab import (
    SafeLabSourceContext,
    validate_safe_lab_source_context,
)
from src.roll2film.density_domain import DensityDomainNegativePrintOperator


class NativeAo6BaseProfileF32V1(ctypes.Structure):
    _MatrixRow = ctypes.c_double * 3
    _Matrix = _MatrixRow * 3
    _DoubleVector = ctypes.c_double * 3
    _FloatVector = ctypes.c_float * 3
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("capture_matrix", _Matrix),
        ("negative_midpoints", _DoubleVector),
        ("negative_slopes", _DoubleVector),
        ("negative_maximum_densities", _DoubleVector),
        ("dye_absorption_matrix", _Matrix),
        ("print_matrix", _Matrix),
        ("paper_midpoints", _DoubleVector),
        ("paper_slopes", _DoubleVector),
        ("paper_maximum_densities", _DoubleVector),
        ("black_endpoint", _DoubleVector),
        ("white_endpoint", _DoubleVector),
        ("destination_mean", _FloatVector),
        ("destination_std", _FloatVector),
        ("exposure_floor", ctypes.c_double),
        ("matrix_minimum_determinant", ctypes.c_double),
        ("minimum_endpoint_span", ctypes.c_double),
        ("density_strength", ctypes.c_double),
        ("style_strength", ctypes.c_float),
        ("luma_strength", ctypes.c_float),
        ("gamut_iterations", ctypes.c_uint32),
        ("output_margin_8bit", ctypes.c_uint32),
    ]


class NativeAo6BaseContextF32V1(ctypes.Structure):
    _Vector = ctypes.c_float * 3
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("pixel_count", ctypes.c_uint64),
        ("source_mean", _Vector),
        ("source_std", _Vector),
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


def native_ao6_base_profile_struct(
    display_payload: dict[str, Any],
) -> NativeAo6BaseProfileF32V1:
    base = display_payload["base"]
    anchor = display_payload["anchor"]
    if (
        base["order"] != "density_then_anchor"
        or float(base["density_strength"]) != 0.5
        or int(base["final_output_margin"]) != 4
        or str(anchor["style"]) != "velvia_50"
        or float(anchor["strength"]) != 0.58
        or float(anchor["luma_strength"]) != 0.35
        or float(anchor["grain"]) != 0.0
        or str(anchor["gamut_mode"]) != "chroma"
    ):
        raise ValueError("native AO6 base contract drift")
    operator = DensityDomainNegativePrintOperator.from_dict(
        base["density_operator"]
    )
    destination_mean = np.asarray(
        anchor["stats"]["mean"], dtype=np.float32
    )
    destination_std = np.asarray(
        anchor["stats"]["std"], dtype=np.float32
    )
    if (
        destination_mean.shape != (3,)
        or destination_std.shape != (3,)
        or not np.all(np.isfinite(destination_mean))
        or not np.all(np.isfinite(destination_std))
        or np.any(destination_std <= 0.0)
    ):
        raise ValueError("native AO6 destination statistics are invalid")
    black, white = operator._raw_endpoints()
    profile = NativeAo6BaseProfileF32V1()
    profile.struct_size = ctypes.sizeof(NativeAo6BaseProfileF32V1)
    profile.abi_version = 1
    profile.source_component_sha256 = _payload_sha256(
        display_payload
    ).encode("ascii")
    for row in range(3):
        for column in range(3):
            profile.capture_matrix[row][column] = float(
                operator.capture_matrix[row, column]
            )
            profile.dye_absorption_matrix[row][column] = float(
                operator.dye_absorption_matrix[row, column]
            )
            profile.print_matrix[row][column] = float(
                operator.print_matrix[row, column]
            )
        profile.negative_midpoints[row] = float(
            operator.negative_midpoints[row]
        )
        profile.negative_slopes[row] = float(
            operator.negative_slopes[row]
        )
        profile.negative_maximum_densities[row] = float(
            operator.negative_maximum_densities[row]
        )
        profile.paper_midpoints[row] = float(
            operator.paper_midpoints[row]
        )
        profile.paper_slopes[row] = float(operator.paper_slopes[row])
        profile.paper_maximum_densities[row] = float(
            operator.paper_maximum_densities[row]
        )
        profile.black_endpoint[row] = float(black[row])
        profile.white_endpoint[row] = float(white[row])
        profile.destination_mean[row] = float(destination_mean[row])
        profile.destination_std[row] = float(destination_std[row])
    profile.exposure_floor = float(operator.exposure_floor)
    profile.matrix_minimum_determinant = float(
        operator.matrix_minimum_determinant
    )
    profile.minimum_endpoint_span = float(
        operator.minimum_endpoint_span
    )
    profile.density_strength = 0.5
    profile.style_strength = np.float32(0.58)
    profile.luma_strength = np.float32(0.35)
    profile.gamut_iterations = 14
    profile.output_margin_8bit = 4
    return profile

def native_ao6_base_context_struct(
    context: SafeLabSourceContext,
) -> NativeAo6BaseContextF32V1:
    validate_safe_lab_source_context(context)
    result = NativeAo6BaseContextF32V1()
    result.struct_size = ctypes.sizeof(NativeAo6BaseContextF32V1)
    result.abi_version = 1
    result.pixel_count = context.pixel_count
    for channel in range(3):
        result.source_mean[channel] = np.float32(
            context.lab_mean[channel]
        )
        result.source_std[channel] = np.float32(
            context.lab_std[channel]
        )
    return result


def native_ao6_base_display_payload_sha256(
    display_payload: dict[str, Any],
) -> str:
    native_ao6_base_profile_struct(display_payload)
    return _payload_sha256(display_payload)
