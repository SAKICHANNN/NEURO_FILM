"""Portable binding for the retained P4BW density-amplitude compiler."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from .density_conditioned_thomas import DensityConditionedThomasProfile
from .manufacturer_characteristic import ManufacturerCharacteristicPrior

MAX_KNOTS = 16


class NativeGranularityAmplitudeError(RuntimeError):
    """Raised when the native P4BW amplitude ABI rejects a request."""


class NativeGranularityAmplitudeProfileV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_profile_sha256", ctypes.c_char * 65),
        ("knot_count", ctypes.c_uint32 * 3),
        ("log_exposure_knots", (ctypes.c_double * MAX_KNOTS) * 3),
        ("density_knots", (ctypes.c_double * MAX_KNOTS) * 3),
        ("channel_floor_variance", ctypes.c_double * 3),
        ("shared_amplitude", ctypes.c_double),
        ("measurement_energy", ctypes.c_double),
    ]


def compile_native_granularity_amplitude_profile(
    bundle: dict[str, object], prior_bundle: dict[str, object]
) -> NativeGranularityAmplitudeProfileV1:
    profile = DensityConditionedThomasProfile.from_dict(bundle)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_bundle["prior"])  # type: ignore[arg-type]
    if profile.amplitude_profile.characteristic_prior_identity != prior.identity():
        raise ValueError("P4BW amplitude and characteristic identities differ")
    result = NativeGranularityAmplitudeProfileV1()
    result.struct_size = ctypes.sizeof(NativeGranularityAmplitudeProfileV1)
    result.abi_version = 1
    result.source_profile_sha256 = profile.identity().encode("ascii")
    for channel, curve in enumerate(prior.curves):
        count = len(curve.log_exposure_knots)
        if count > MAX_KNOTS:
            raise ValueError("P4BW characteristic exceeds native knot capacity")
        result.knot_count[channel] = count
        for index in range(count):
            result.log_exposure_knots[channel][index] = float(
                curve.log_exposure_knots[index]
            )
            result.density_knots[channel][index] = float(curve.density_knots[index])
        layer = ("red", "green", "blue")[channel]
        result.channel_floor_variance[channel] = float(
            profile.amplitude_profile.channel_floor_variance[layer]
        )
    result.shared_amplitude = profile.amplitude_profile.shared_amplitude
    result.measurement_energy = profile.measurement_energy()
    return result


def load_native_granularity_amplitude_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    library.nf_granularity_amplitude_f32_abi_version_v1.argtypes = []
    library.nf_granularity_amplitude_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_granularity_amplitude_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeGranularityAmplitudeProfileV1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_granularity_amplitude_f32_apply_v1.restype = ctypes.c_int
    library.nf_granularity_amplitude_f32_apply_layer_v1.argtypes = [
        ctypes.POINTER(NativeGranularityAmplitudeProfileV1),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_granularity_amplitude_f32_apply_layer_v1.restype = ctypes.c_int
    if library.nf_granularity_amplitude_f32_abi_version_v1() != 1:
        raise NativeGranularityAmplitudeError("native amplitude ABI version mismatch")
    return library


def apply_native_granularity_amplitude(
    library: ctypes.CDLL,
    profile: NativeGranularityAmplitudeProfileV1,
    relative_log_exposure_chw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    exposure = np.asarray(relative_log_exposure_chw)
    if (
        exposure.dtype != np.float32
        or exposure.ndim < 2
        or exposure.shape[0] != 3
        or not exposure.flags.c_contiguous
    ):
        raise ValueError("relative log exposure must be contiguous float32 CHW")
    density = np.empty_like(exposure)
    sigma = np.empty_like(exposure)
    pointer = ctypes.POINTER(ctypes.c_float)
    status = library.nf_granularity_amplitude_f32_apply_v1(
        ctypes.byref(profile),
        exposure.ctypes.data_as(pointer),
        exposure[0].size,
        density.ctypes.data_as(pointer),
        sigma.ctypes.data_as(pointer),
    )
    if status != 0:
        raise NativeGranularityAmplitudeError(
            f"native granularity amplitude failed: {status}"
        )
    return density, sigma


__all__ = [
    "NativeGranularityAmplitudeError",
    "NativeGranularityAmplitudeProfileV1",
    "apply_native_granularity_amplitude",
    "compile_native_granularity_amplitude_profile",
    "load_native_granularity_amplitude_library",
]
