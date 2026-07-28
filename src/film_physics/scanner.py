"""Explicit scanner nuisance/output-profile primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

from .structure_compiler import counter_normal_region


SCANNER_STAGES = ("spectral", "flare", "dmax", "mtf", "noise")


@dataclass(frozen=True)
class ScannerProfile:
    profile_id: str
    illuminant_rgb: tuple[float, float, float]
    spectral_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    local_flare_fraction: float
    global_flare_fraction: float
    flare_sigma_um: float
    dmax_density_rgb: tuple[float, float, float] | None
    mtf_sigma_um_rgb: tuple[float, float, float]
    shot_noise_variance_scale: float
    read_noise_variance: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("scanner profile_id must be non-empty")
        if len(self.illuminant_rgb) != 3 or any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in self.illuminant_rgb
        ):
            raise ValueError("scanner illuminant must contain three values in (0, 1]")
        matrix = np.asarray(self.spectral_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("scanner spectral matrix must be finite 3x3")
        if np.any(matrix < 0.0) or np.any(np.sum(matrix, axis=1) > 1.0):
            raise ValueError("scanner spectral rows must be nonnegative and sum <= 1")
        fractions = (self.local_flare_fraction, self.global_flare_fraction)
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise ValueError("scanner flare fractions must be finite and nonnegative")
        if sum(fractions) >= 1.0:
            raise ValueError("scanner flare fractions must sum below one")
        if not math.isfinite(self.flare_sigma_um) or self.flare_sigma_um < 0.0:
            raise ValueError("scanner flare sigma must be finite and nonnegative")
        if self.local_flare_fraction > 0.0 and self.flare_sigma_um == 0.0:
            raise ValueError("nonzero local flare requires a nonzero sigma")
        if self.dmax_density_rgb is not None and (
            len(self.dmax_density_rgb) != 3
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.dmax_density_rgb
            )
        ):
            raise ValueError("scanner Dmax must contain three positive densities")
        if len(self.mtf_sigma_um_rgb) != 3 or any(
            not math.isfinite(value) or value < 0.0
            for value in self.mtf_sigma_um_rgb
        ):
            raise ValueError("scanner MTF sigmas must be finite and nonnegative")
        noise = (self.shot_noise_variance_scale, self.read_noise_variance)
        if any(not math.isfinite(value) or value < 0.0 for value in noise):
            raise ValueError("scanner noise variances must be finite and nonnegative")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**64 - 3:
            raise ValueError("scanner seed must leave room for three channels")


@dataclass(frozen=True)
class ScannerContext:
    profile_id: str
    full_shape: tuple[int, int]
    stages: tuple[str, ...]
    global_spectral_mean_rgb: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("scanner context profile_id must be non-empty")
        if (
            len(self.full_shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.full_shape
            )
        ):
            raise ValueError("scanner context full_shape must be positive HxW")
        object.__setattr__(self, "stages", _validate_stages(self.stages))
        if len(self.global_spectral_mean_rgb) != 3 or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.global_spectral_mean_rgb
        ):
            raise ValueError("scanner context mean must contain three bounded values")

    @property
    def context_id(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "full_shape": list(self.full_shape),
            "stages": list(self.stages),
            "global_spectral_mean_rgb": list(self.global_spectral_mean_rgb),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()


@dataclass(frozen=True)
class ScannerStandardContext:
    profile_id: str
    full_shape: tuple[int, int]
    stages: tuple[str, ...]
    global_spectral_mean_rgb: tuple[float, float, float]
    arithmetic: str = "float32-standard"

    def __post_init__(self) -> None:
        if self.arithmetic != "float32-standard":
            raise ValueError("unsupported scanner Standard arithmetic")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("scanner Standard context profile_id must be non-empty")
        if (
            len(self.full_shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.full_shape
            )
        ):
            raise ValueError("scanner Standard full_shape must be positive HxW")
        object.__setattr__(self, "stages", _validate_stages(self.stages))
        if len(self.global_spectral_mean_rgb) != 3 or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.global_spectral_mean_rgb
        ):
            raise ValueError("scanner Standard mean must be bounded")

    @property
    def context_id(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "full_shape": list(self.full_shape),
            "stages": list(self.stages),
            "global_spectral_mean_rgb": list(self.global_spectral_mean_rgb),
            "arithmetic": self.arithmetic,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()


def _validate_transmittance(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("scanner input must be HxWx3")
    if (
        not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError("scanner transmittance must be finite in (0, 1]")
    return array


def _validate_standard_transmittance(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise TypeError("scanner Standard input must be float32")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("scanner input must be HxWx3")
    if (
        not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError("scanner transmittance must be finite in (0, 1]")
    return array


def _validate_stages(stages: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(stages)
    if len(selected) != len(set(selected)):
        raise ValueError("scanner stages must be unique")
    if any(stage not in SCANNER_STAGES for stage in selected):
        raise ValueError("unsupported scanner stage")
    if selected != tuple(stage for stage in SCANNER_STAGES if stage in selected):
        raise ValueError("scanner stages must preserve canonical order")
    return selected


def _blur_rgb(
    values: np.ndarray,
    sigmas_um: tuple[float, float, float],
    pixel_pitch_um: float,
) -> np.ndarray:
    output = np.empty_like(values)
    for channel, sigma_um in enumerate(sigmas_um):
        sigma = sigma_um / pixel_pitch_um
        if sigma == 0.0:
            output[..., channel] = values[..., channel]
        else:
            output[..., channel] = gaussian_filter(
                values[..., channel],
                sigma=sigma,
                mode="nearest",
                truncate=4.0,
            )
    return output


def _blur_rgb_standard(
    values: np.ndarray,
    sigmas_um: tuple[float, float, float],
    pixel_pitch_um: float,
) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for channel, sigma_um in enumerate(sigmas_um):
        sigma = float(sigma_um / pixel_pitch_um)
        if sigma == 0.0:
            output[..., channel] = values[..., channel]
        else:
            gaussian_filter(
                values[..., channel],
                sigma=sigma,
                mode="nearest",
                truncate=4.0,
                output=output[..., channel],
            )
    return output


def _smooth_bounded_noise(
    signal: np.ndarray,
    profile: ScannerProfile,
    *,
    full_shape: tuple[int, int],
    origin_yx: tuple[int, int],
) -> np.ndarray:
    variance = (
        profile.shot_noise_variance_scale * signal
        + profile.read_noise_variance
    )
    raw = np.empty_like(signal)
    for channel in range(3):
        normal = counter_normal_region(
            full_shape,
            origin_yx=origin_yx,
            shape=signal.shape[:2],
            seed=profile.seed + channel,
        )
        raw[..., channel] = np.sqrt(variance[..., channel]) * normal
    limit = np.where(raw >= 0.0, 1.0 - signal, signal)
    correction = np.zeros_like(raw)
    active = limit > 0.0
    correction[active] = (
        np.sign(raw[active])
        * limit[active]
        * np.tanh(np.abs(raw[active]) / limit[active])
    )
    return signal + correction


def compile_scanner_context(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    stages: Iterable[str] = SCANNER_STAGES,
) -> ScannerContext:
    values = _validate_transmittance(transmittance)
    selected = _validate_stages(stages)
    mean = np.mean(values, axis=(0, 1))
    if "spectral" in selected:
        mean = (
            mean * np.asarray(profile.illuminant_rgb)
        ) @ np.asarray(profile.spectral_matrix).T
    return ScannerContext(
        profile_id=profile.profile_id,
        full_shape=values.shape[:2],
        stages=selected,
        global_spectral_mean_rgb=tuple(float(value) for value in mean),
    )


def required_scanner_halo(
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    stages: Iterable[str] = SCANNER_STAGES,
) -> int:
    if not math.isfinite(pixel_pitch_um) or pixel_pitch_um <= 0.0:
        raise ValueError("scanner pixel pitch must be finite and positive")
    selected = _validate_stages(stages)
    flare_radius = (
        int(4.0 * profile.flare_sigma_um / pixel_pitch_um + 0.5)
        if "flare" in selected and profile.local_flare_fraction > 0.0
        else 0
    )
    mtf_radius = (
        int(4.0 * max(profile.mtf_sigma_um_rgb) / pixel_pitch_um + 0.5)
        if "mtf" in selected
        else 0
    )
    return flare_radius + mtf_radius


def apply_scanner_profile(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    stages: Iterable[str] = SCANNER_STAGES,
    full_shape: tuple[int, int] | None = None,
    origin_yx: tuple[int, int] = (0, 0),
    context: ScannerContext | None = None,
) -> np.ndarray:
    """Map film transmittance to a bounded relative scanner-linear signal."""
    values = _validate_transmittance(transmittance)
    if not math.isfinite(pixel_pitch_um) or pixel_pitch_um <= 0.0:
        raise ValueError("scanner pixel pitch must be finite and positive")
    selected = _validate_stages(stages)
    if context is not None:
        if context.profile_id != profile.profile_id:
            raise ValueError("scanner context profile mismatch")
        if context.stages != selected:
            raise ValueError("scanner context stage mismatch")
        if full_shape is not None and tuple(full_shape) != context.full_shape:
            raise ValueError("scanner context full_shape mismatch")
        full_shape = context.full_shape
    if full_shape is None:
        full_shape = values.shape[:2]
    if not (
        0 <= origin_yx[0] < origin_yx[0] + values.shape[0] <= full_shape[0]
        and 0 <= origin_yx[1] < origin_yx[1] + values.shape[1] <= full_shape[1]
    ):
        raise ValueError("scanner region is outside full_shape")
    signal = values
    if "spectral" in selected:
        illuminated = signal * np.asarray(profile.illuminant_rgb)
        signal = illuminated @ np.asarray(profile.spectral_matrix).T
    if "flare" in selected:
        local = profile.local_flare_fraction
        global_fraction = profile.global_flare_fraction
        if local > 0.0:
            blurred = _blur_rgb(
                signal,
                (profile.flare_sigma_um,) * 3,
                pixel_pitch_um,
            )
        else:
            blurred = signal
        mean = (
            np.asarray(context.global_spectral_mean_rgb).reshape(1, 1, 3)
            if context is not None
            else np.mean(signal, axis=(0, 1), keepdims=True)
        )
        signal = (
            (1.0 - local - global_fraction) * signal
            + local * blurred
            + global_fraction * mean
        )
    if "dmax" in selected and profile.dmax_density_rgb is not None:
        floor = np.power(10.0, -np.asarray(profile.dmax_density_rgb))
        signal = floor + (1.0 - floor) * signal
    if "mtf" in selected:
        signal = _blur_rgb(signal, profile.mtf_sigma_um_rgb, pixel_pitch_um)
    if "noise" in selected and (
        profile.shot_noise_variance_scale > 0.0
        or profile.read_noise_variance > 0.0
    ):
        signal = _smooth_bounded_noise(
            signal,
            profile,
            full_shape=full_shape,
            origin_yx=origin_yx,
        )
    if (
        not np.all(np.isfinite(signal))
        or np.any(signal < 0.0)
        or np.any(signal > 1.0)
    ):
        raise RuntimeError("scanner profile left bounded scan-linear domain")
    return signal


def apply_scanner_profile_row_tiled(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    context: ScannerContext,
    tile_rows: int,
    stages: Iterable[str] = SCANNER_STAGES,
) -> np.ndarray:
    values = _validate_transmittance(transmittance)
    selected = _validate_stages(stages)
    if context.full_shape != values.shape[:2]:
        raise ValueError("scanner context does not bind this full input shape")
    if context.profile_id != profile.profile_id or context.stages != selected:
        raise ValueError("scanner context does not bind profile/stages")
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    halo = required_scanner_halo(
        profile, pixel_pitch_um=pixel_pitch_um, stages=selected
    )
    output = np.empty_like(values)
    for y0 in range(0, values.shape[0], tile_rows):
        y1 = min(values.shape[0], y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(values.shape[0], y1 + halo)
        rendered = apply_scanner_profile(
            values[source_y0:source_y1],
            profile,
            pixel_pitch_um=pixel_pitch_um,
            stages=selected,
            full_shape=values.shape[:2],
            origin_yx=(source_y0, 0),
            context=context,
        )
        output[y0:y1] = rendered[y0 - source_y0 : y1 - source_y0]
    return output


def compile_scanner_standard_context(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    stages: Iterable[str] = SCANNER_STAGES,
) -> ScannerStandardContext:
    values = _validate_standard_transmittance(transmittance)
    selected = _validate_stages(stages)
    mean = np.mean(values, axis=(0, 1), dtype=np.float64)
    if "spectral" in selected:
        mean = (
            mean * np.asarray(profile.illuminant_rgb, dtype=np.float64)
        ) @ np.asarray(profile.spectral_matrix, dtype=np.float64).T
    mean32 = np.asarray(mean, dtype=np.float32)
    return ScannerStandardContext(
        profile_id=profile.profile_id,
        full_shape=values.shape[:2],
        stages=selected,
        global_spectral_mean_rgb=tuple(float(value) for value in mean32),
    )


def _apply_scanner_profile_standard_region(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    context: ScannerStandardContext,
    origin_yx: tuple[int, int],
) -> np.ndarray:
    values = _validate_standard_transmittance(transmittance)
    selected = context.stages
    signal = values
    if "spectral" in selected:
        illuminated = signal * np.asarray(
            profile.illuminant_rgb, dtype=np.float32
        )
        signal = illuminated @ np.asarray(
            profile.spectral_matrix, dtype=np.float32
        ).T
    if "flare" in selected:
        local = np.float32(profile.local_flare_fraction)
        global_fraction = np.float32(profile.global_flare_fraction)
        blurred = (
            _blur_rgb_standard(
                signal,
                (profile.flare_sigma_um,) * 3,
                pixel_pitch_um,
            )
            if local > 0.0
            else signal
        )
        mean = np.asarray(
            context.global_spectral_mean_rgb, dtype=np.float32
        ).reshape(1, 1, 3)
        signal = (
            (np.float32(1.0) - local - global_fraction) * signal
            + local * blurred
            + global_fraction * mean
        ).astype(np.float32, copy=False)
    if "dmax" in selected and profile.dmax_density_rgb is not None:
        floor = np.power(
            np.float32(10.0),
            -np.asarray(profile.dmax_density_rgb, dtype=np.float32),
        )
        signal = (
            floor + (np.float32(1.0) - floor) * signal
        ).astype(np.float32, copy=False)
    if "mtf" in selected:
        signal = _blur_rgb_standard(
            signal, profile.mtf_sigma_um_rgb, pixel_pitch_um
        )
    if "noise" in selected and (
        profile.shot_noise_variance_scale > 0.0
        or profile.read_noise_variance > 0.0
    ):
        output = np.array(signal, copy=True, dtype=np.float32)
        shot = np.float32(profile.shot_noise_variance_scale)
        read = np.float32(profile.read_noise_variance)
        for channel in range(3):
            normal = counter_normal_region(
                context.full_shape,
                origin_yx=origin_yx,
                shape=signal.shape[:2],
                seed=profile.seed + channel,
            ).astype(np.float32)
            variance = shot * signal[..., channel] + read
            np.sqrt(variance, out=variance)
            raw = variance * normal
            limit = np.where(
                raw >= 0.0,
                np.float32(1.0) - signal[..., channel],
                signal[..., channel],
            )
            correction = np.zeros_like(raw)
            active = limit > 0.0
            correction[active] = (
                np.sign(raw[active])
                * limit[active]
                * np.tanh(np.abs(raw[active]) / limit[active])
            )
            output[..., channel] += correction
        signal = output
    if (
        signal.dtype != np.float32
        or not np.all(np.isfinite(signal))
        or np.any(signal < 0.0)
        or np.any(signal > 1.0)
    ):
        raise RuntimeError("scanner Standard left bounded float32 domain")
    return signal


def apply_scanner_profile_standard_row_tiled(
    transmittance: np.ndarray,
    profile: ScannerProfile,
    *,
    pixel_pitch_um: float,
    context: ScannerStandardContext,
    tile_rows: int,
) -> np.ndarray:
    values = _validate_standard_transmittance(transmittance)
    if context.profile_id != profile.profile_id:
        raise ValueError("scanner Standard context profile mismatch")
    if context.full_shape != values.shape[:2]:
        raise ValueError("scanner Standard context shape mismatch")
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
    ):
        raise ValueError("tile_rows must be a positive integer")
    halo = required_scanner_halo(
        profile, pixel_pitch_um=pixel_pitch_um, stages=context.stages
    )
    output = np.empty_like(values, dtype=np.float32)
    for y0 in range(0, values.shape[0], tile_rows):
        y1 = min(values.shape[0], y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(values.shape[0], y1 + halo)
        rendered = _apply_scanner_profile_standard_region(
            values[source_y0:source_y1],
            profile,
            pixel_pitch_um=pixel_pitch_um,
            context=context,
            origin_yx=(source_y0, 0),
        )
        output[y0:y1] = rendered[y0 - source_y0 : y1 - source_y0]
    return output
