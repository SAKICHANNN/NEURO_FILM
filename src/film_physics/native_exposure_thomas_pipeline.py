"""Serial native exposure-to-density-to-Thomas composition."""

from __future__ import annotations

import ctypes

import numpy as np

from .native_granularity_amplitude import (
    NativeGranularityAmplitudeError,
    NativeGranularityAmplitudeProfileV1,
)
from .native_thomas_density import NativeThomasDensityError
from .native_thomas_field import NativeThomasFieldProfile


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def render_native_exposure_thomas_rgb(
    amplitude_library: ctypes.CDLL,
    thomas_library: ctypes.CDLL,
    amplitude_profile: NativeGranularityAmplitudeProfileV1,
    field_profiles: tuple[
        NativeThomasFieldProfile,
        NativeThomasFieldProfile,
        NativeThomasFieldProfile,
    ],
    relative_log_exposure_chw: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float], int]:
    """Compile and render one layer at a time with shared intermediates."""

    exposure = np.asarray(relative_log_exposure_chw)
    if (
        exposure.dtype != np.float32
        or exposure.ndim != 3
        or exposure.shape[0] != 3
        or not exposure.flags.c_contiguous
    ):
        raise ValueError("relative log exposure must be contiguous float32 CHW")
    for channel in range(3):
        values = exposure[channel]
        count = int(amplitude_profile.knot_count[channel])
        lower = float(amplitude_profile.log_exposure_knots[channel][0])
        upper = float(amplitude_profile.log_exposure_knots[channel][count - 1])
        if (
            not np.all(np.isfinite(values))
            or float(np.min(values)) < lower
            or float(np.max(values)) > upper
        ):
            raise ValueError("relative layer log exposure is outside the profile domain")

    _, height, width = exposure.shape
    sample_count = height * width
    workspace_count = ctypes.c_size_t()
    status = thomas_library.nf_thomas_density_f32_workspace_floats_v1(
        height, width, ctypes.byref(workspace_count)
    )
    if status != 0:
        raise NativeThomasDensityError(f"native workspace request failed: {status}")
    density = np.empty(sample_count, dtype=np.float32)
    point_sigma = np.empty(sample_count, dtype=np.float32)
    workspace = np.empty(workspace_count.value, dtype=np.float32)
    output = np.empty_like(exposure)
    raw_means: list[float] = []
    for channel, field_profile in enumerate(field_profiles):
        status = amplitude_library.nf_granularity_amplitude_f32_apply_layer_v1(
            ctypes.byref(amplitude_profile),
            channel,
            _pointer(exposure[channel]),
            sample_count,
            _pointer(density),
            _pointer(point_sigma),
        )
        if status != 0:
            raise NativeGranularityAmplitudeError(
                f"native layer amplitude compilation failed at channel {channel}: {status}"
            )
        raw_mean = ctypes.c_double()
        abi_profile = field_profile.as_abi()
        status = thomas_library.nf_thomas_density_f32_apply_v1(
            ctypes.byref(abi_profile),
            height,
            width,
            _pointer(density),
            sample_count,
            _pointer(point_sigma),
            sample_count,
            _pointer(workspace),
            workspace.size,
            _pointer(output[channel]),
            sample_count,
            ctypes.byref(raw_mean),
        )
        if status != 0:
            raise NativeThomasDensityError(
                f"native exposure-to-Thomas composition failed at channel {channel}: {status}"
            )
        raw_means.append(raw_mean.value)
    output.setflags(write=False)
    reused_bytes = density.nbytes + point_sigma.nbytes + workspace.nbytes
    return output, tuple(raw_means), reused_bytes  # type: ignore[return-value]


__all__ = ["render_native_exposure_thomas_rgb"]
