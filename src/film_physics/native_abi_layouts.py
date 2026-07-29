"""Shared ctypes layouts for the versioned native film-physics ABIs."""

from __future__ import annotations

import ctypes
from typing import Any

from .native_profile import (
    NATIVE_PRINT_ABI_VERSION,
    NATIVE_PRINT_MAX_KNOTS,
)
from .native_spatial_profile import NATIVE_GAUSSIAN_ABI_VERSION


class NativePrintProfileV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("reference_linear", ctypes.c_double),
        ("black_offset", ctypes.c_double),
        ("knot_count", ctypes.c_uint32 * 3),
        (
            "x_knots",
            (ctypes.c_double * NATIVE_PRINT_MAX_KNOTS) * 3,
        ),
        (
            "y_knots",
            (ctypes.c_double * NATIVE_PRINT_MAX_KNOTS) * 3,
        ),
        (
            "derivatives",
            (ctypes.c_double * NATIVE_PRINT_MAX_KNOTS) * 3,
        ),
        ("dye_absorption_matrix", (ctypes.c_double * 3) * 3),
        ("print_matrix", (ctypes.c_double * 3) * 3),
        ("paper_midpoints", ctypes.c_double * 3),
        ("paper_slopes", ctypes.c_double * 3),
        ("paper_maximum_densities", ctypes.c_double * 3),
        ("black_reference_density", ctypes.c_double * 3),
        ("white_reference_density", ctypes.c_double * 3),
        ("exposure_floor", ctypes.c_double),
        ("matrix_minimum_determinant", ctypes.c_double),
    ]


class NativeGaussianProfileV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("sigma_pixels_rgb", ctypes.c_double * 3),
        ("truncate", ctypes.c_double),
    ]


class NativeAdjacencyProfileV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_component_sha256", ctypes.c_char * 65),
        ("gain_rgb", ctypes.c_double * 3),
        ("maximum_absolute_transmittance_delta", ctypes.c_double),
        ("maximum_absolute_density_delta", ctypes.c_double),
        ("black_reference_density", ctypes.c_double * 3),
        ("white_reference_density", ctypes.c_double * 3),
    ]


def native_print_profile_struct(
    payload: dict[str, Any],
) -> NativePrintProfileV1:
    operator = payload["operator"]
    sensitometry = operator["sensitometry"]
    interpretation = operator["interpretation"]
    result = NativePrintProfileV1()
    result.struct_size = ctypes.sizeof(NativePrintProfileV1)
    result.abi_version = NATIVE_PRINT_ABI_VERSION
    result.source_component_sha256 = payload["source_component"][
        "sha256"
    ].encode("ascii")
    result.reference_linear = float(
        sensitometry["encoder"]["reference_linear"]
    )
    result.black_offset = float(
        sensitometry["encoder"]["black_offset"]
    )
    for channel, curve in enumerate(sensitometry["curves"]):
        spline = curve["spline"]
        count = len(spline["x_knots"])
        result.knot_count[channel] = count
        for index in range(count):
            result.x_knots[channel][index] = float(
                spline["x_knots"][index]
            )
            result.y_knots[channel][index] = float(
                spline["y_knots"][index]
            )
            result.derivatives[channel][index] = float(
                spline["derivatives"][index]
            )
    _copy_matrix(
        result.dye_absorption_matrix,
        interpretation["dye_absorption_matrix"],
    )
    _copy_matrix(result.print_matrix, interpretation["print_matrix"])
    for name in (
        "paper_midpoints",
        "paper_slopes",
        "paper_maximum_densities",
        "black_reference_density",
        "white_reference_density",
    ):
        target = getattr(result, name)
        for index, value in enumerate(interpretation[name]):
            target[index] = float(value)
    result.exposure_floor = float(interpretation["exposure_floor"])
    result.matrix_minimum_determinant = float(
        interpretation["matrix_minimum_determinant"]
    )
    return result


def native_gaussian_profile_struct(
    payload: dict[str, Any],
    stage: dict[str, Any],
) -> NativeGaussianProfileV1:
    profile = NativeGaussianProfileV1()
    profile.struct_size = ctypes.sizeof(NativeGaussianProfileV1)
    profile.abi_version = NATIVE_GAUSSIAN_ABI_VERSION
    profile.source_component_sha256 = payload["source_component"][
        "sha256"
    ].encode("ascii")
    for channel, sigma in enumerate(stage["sigma_pixels_rgb"]):
        profile.sigma_pixels_rgb[channel] = float(sigma)
    profile.truncate = float(payload["gaussian_truncate"])
    return profile


def native_adjacency_profile_struct(
    payload: dict[str, Any],
) -> NativeAdjacencyProfileV1:
    profile = NativeAdjacencyProfileV1()
    profile.struct_size = ctypes.sizeof(NativeAdjacencyProfileV1)
    profile.abi_version = 1
    profile.source_component_sha256 = payload["source_component"][
        "sha256"
    ].encode("ascii")
    for channel in range(3):
        profile.gain_rgb[channel] = float(
            payload["development_adjacency_gain_rgb"][channel]
        )
        profile.black_reference_density[channel] = float(
            payload["black_reference_density"][channel]
        )
        profile.white_reference_density[channel] = float(
            payload["white_reference_density"][channel]
        )
    profile.maximum_absolute_transmittance_delta = float(
        payload["maximum_absolute_transmittance_delta"]
    )
    profile.maximum_absolute_density_delta = float(
        payload["maximum_absolute_density_delta"]
    )
    return profile


def _copy_matrix(target: Any, values: list[list[float]]) -> None:
    for row in range(3):
        for column in range(3):
            target[row][column] = float(values[row][column])


__all__ = [
    "NativeAdjacencyProfileV1",
    "NativeGaussianProfileV1",
    "NativePrintProfileV1",
    "native_adjacency_profile_struct",
    "native_gaussian_profile_struct",
    "native_print_profile_struct",
]
