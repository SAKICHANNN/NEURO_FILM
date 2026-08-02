"""Finite-crystal exposure capture with explicit remaining-light depletion.

This is a clean-room synthetic mechanism, not a calibrated emulsion model.
Crystal footprints are finite, deterministic and non-overlapping after a
canonical nearest-centre ownership rule.  A crystal captures one flat value
derived from the mean exposure still available over its footprint; the next
layer receives only the remaining exposure.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit
from .structure_compiler import counter_uniform_region

CRYSTAL_LIGHT_DEPLETION_SCHEMA = "neuro_film.crystal_light_depletion_profile.v1"


class CrystalLightDepletionError(ValueError):
    """Raised when crystal geometry or exposure capture is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class CrystalLightDepletionProfile:
    profile_id: str
    layer_count_per_channel: int
    cell_pitch_pixels: int
    activation_probability: float
    radius_min_pixels: float
    radius_max_pixels: float
    centre_jitter_fraction: float
    capture_efficiency: float
    channel_seed_bases: tuple[int, int, int]
    layer_seed_stride: int

    def __post_init__(self) -> None:
        probabilities = (self.activation_probability, self.capture_efficiency)
        radii = (self.radius_min_pixels, self.radius_max_pixels)
        if (
            not self.profile_id
            or not isinstance(self.layer_count_per_channel, int)
            or not 1 <= self.layer_count_per_channel <= 64
            or not isinstance(self.cell_pitch_pixels, int)
            or self.cell_pitch_pixels < 2
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0
                for value in probabilities
            )
            or any(not math.isfinite(value) or value <= 0.0 for value in radii)
            or self.radius_min_pixels > self.radius_max_pixels
            or self.radius_max_pixels >= self.cell_pitch_pixels
            or not math.isfinite(self.centre_jitter_fraction)
            or not 0.0 <= self.centre_jitter_fraction <= 1.0
            or len(self.channel_seed_bases) != 3
            or any(
                not isinstance(value, int) or not 0 <= value < 2**64
                for value in self.channel_seed_bases
            )
            or not isinstance(self.layer_seed_stride, int)
            or not 0 < self.layer_seed_stride < 2**64
        ):
            raise CrystalLightDepletionError("invalid crystal depletion profile")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": CRYSTAL_LIGHT_DEPLETION_SCHEMA,
            "profile_id": self.profile_id,
            "layer_count_per_channel": self.layer_count_per_channel,
            "cell_pitch_pixels": self.cell_pitch_pixels,
            "activation_probability": self.activation_probability,
            "radius_min_pixels": self.radius_min_pixels,
            "radius_max_pixels": self.radius_max_pixels,
            "centre_jitter_fraction": self.centre_jitter_fraction,
            "capture_efficiency": self.capture_efficiency,
            "channel_seed_bases": list(self.channel_seed_bases),
            "layer_seed_stride": self.layer_seed_stride,
        }

    def identity(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.identity_payload())).hexdigest()


@dataclass(frozen=True)
class CrystalFootprintMap:
    labels: np.ndarray
    crystal_count: int
    coverage_fraction: float
    geometry_id: str

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if (
            labels.ndim != 2
            or labels.dtype != np.int32
            or labels.size == 0
            or self.crystal_count < 1
            or not math.isfinite(self.coverage_fraction)
            or not 0.0 < self.coverage_fraction <= 1.0
            or not self.geometry_id
        ):
            raise CrystalLightDepletionError("invalid crystal footprint map")
        owned = np.array(labels, copy=True, order="C")
        owned.setflags(write=False)
        object.__setattr__(self, "labels", owned)


@dataclass(frozen=True)
class CrystalCaptureResult:
    captured_exposure: PhysicalDomainArray
    remaining_exposure: PhysicalDomainArray
    layer_captures: tuple[np.ndarray, ...]
    geometry_ids: tuple[tuple[str, str, str], ...]
    coverage_fractions: tuple[tuple[float, float, float], ...]
    profile_id: str


def _layer_seed(profile: CrystalLightDepletionProfile, channel: int, layer: int) -> int:
    return int(
        (profile.channel_seed_bases[channel] + layer * profile.layer_seed_stride)
        % 2**64
    )


def build_crystal_footprint_map(
    shape: tuple[int, int],
    profile: CrystalLightDepletionProfile,
    *,
    channel_index: int,
    layer_index: int,
) -> CrystalFootprintMap:
    """Build one deterministic finite-footprint layer in canonical order."""
    if (
        len(shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in shape)
        or not 0 <= channel_index < 3
        or not 0 <= layer_index < profile.layer_count_per_channel
    ):
        raise CrystalLightDepletionError("invalid crystal footprint request")
    height, width = shape
    pitch = profile.cell_pitch_pixels
    grid_shape = (math.ceil(height / pitch) + 2, math.ceil(width / pitch) + 2)
    seed = _layer_seed(profile, channel_index, layer_index)
    uniforms = [
        counter_uniform_region(
            grid_shape,
            origin_yx=(0, 0),
            shape=grid_shape,
            seed=seed ^ salt,
        )
        for salt in (
            0x9E3779B97F4A7C15,
            0xD1B54A32D192ED03,
            0x94D049BB133111EB,
            0x8538ECB5BD456EA3,
        )
    ]
    labels = np.full(shape, -1, dtype=np.int32)
    best = np.full(shape, np.inf, dtype=np.float64)
    crystal_id = 0
    jitter_scale = profile.centre_jitter_fraction * pitch
    radius_span = profile.radius_max_pixels - profile.radius_min_pixels
    for grid_y in range(grid_shape[0]):
        for grid_x in range(grid_shape[1]):
            if uniforms[0][grid_y, grid_x] >= profile.activation_probability:
                continue
            centre_y = (
                (grid_y - 1) * pitch
                + 0.5 * pitch
                + (uniforms[1][grid_y, grid_x] - 0.5) * jitter_scale
            )
            centre_x = (
                (grid_x - 1) * pitch
                + 0.5 * pitch
                + (uniforms[2][grid_y, grid_x] - 0.5) * jitter_scale
            )
            radius = (
                profile.radius_min_pixels + uniforms[3][grid_y, grid_x] * radius_span
            )
            y0 = max(0, math.floor(centre_y - radius))
            y1 = min(height, math.floor(centre_y + radius) + 1)
            x0 = max(0, math.floor(centre_x - radius))
            x1 = min(width, math.floor(centre_x + radius) + 1)
            if y0 < y1 and x0 < x1:
                ys = np.arange(y0, y1, dtype=np.float64)[:, None]
                xs = np.arange(x0, x1, dtype=np.float64)[None, :]
                normalized = np.square((ys - centre_y) / radius) + np.square(
                    (xs - centre_x) / radius
                )
                current_best = best[y0:y1, x0:x1]
                selected = (normalized <= 1.0) & (normalized < current_best)
                current_best[selected] = normalized[selected]
                labels[y0:y1, x0:x1][selected] = crystal_id
            crystal_id += 1
    covered = labels >= 0
    if crystal_id == 0 or not np.any(covered):
        raise CrystalLightDepletionError("crystal layer has no covered samples")
    payload = {
        "profile_id": profile.identity(),
        "shape": list(shape),
        "channel_index": channel_index,
        "layer_index": layer_index,
        "seed": seed,
        "labels_sha256": hashlib.sha256(
            np.ascontiguousarray(labels, dtype="<i4").tobytes()
        ).hexdigest(),
    }
    return CrystalFootprintMap(
        labels=labels,
        crystal_count=crystal_id,
        coverage_fraction=float(np.mean(covered, dtype=np.float64)),
        geometry_id=hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    )


def _footprint_capture(
    exposure: np.ndarray, geometry: CrystalFootprintMap, efficiency: float
) -> np.ndarray:
    covered = geometry.labels >= 0
    labels = geometry.labels[covered]
    sums = np.bincount(
        labels,
        weights=np.asarray(exposure[covered], dtype=np.float64),
        minlength=geometry.crystal_count,
    )
    counts = np.bincount(labels, minlength=geometry.crystal_count)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    capture = np.zeros_like(exposure, dtype=np.float64)
    capture[covered] = efficiency * np.minimum(means[labels], exposure[covered])
    return capture


def _validate_exposure(exposure: PhysicalDomainArray) -> None:
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float64 or np.any(exposure.values <= 0.0):
        raise CrystalLightDepletionError(
            "crystal reference execution requires positive float64 layer exposure"
        )


def _wrap_exposure(
    values: np.ndarray, source: PhysicalDomainArray
) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        source.channels,
        source.scale,
    )


def render_crystal_light_depletion(
    exposure: PhysicalDomainArray,
    profile: CrystalLightDepletionProfile,
    *,
    reverse_layers: bool = False,
    point_capture: bool = False,
) -> CrystalCaptureResult:
    """Render footprint-averaged or pointwise capture with remaining-light recursion."""
    _validate_exposure(exposure)
    shape = exposure.values.shape[:2]
    remaining = np.array(exposure.values, dtype=np.float64, copy=True)
    captures: list[np.ndarray] = []
    geometry_ids: list[tuple[str, str, str]] = []
    coverage_fractions: list[tuple[float, float, float]] = []
    order = (
        range(profile.layer_count_per_channel - 1, -1, -1)
        if reverse_layers
        else range(profile.layer_count_per_channel)
    )
    for layer_index in order:
        layer_capture = np.zeros_like(remaining)
        ids: list[str] = []
        coverage: list[float] = []
        for channel_index in range(3):
            geometry = build_crystal_footprint_map(
                shape,
                profile,
                channel_index=channel_index,
                layer_index=layer_index,
            )
            if point_capture:
                selected = geometry.labels >= 0
                channel_capture = np.zeros(shape, dtype=np.float64)
                channel_capture[selected] = (
                    profile.capture_efficiency * remaining[..., channel_index][selected]
                )
            else:
                channel_capture = _footprint_capture(
                    remaining[..., channel_index], geometry, profile.capture_efficiency
                )
            layer_capture[..., channel_index] = channel_capture
            ids.append(geometry.geometry_id)
            coverage.append(geometry.coverage_fraction)
        remaining -= layer_capture
        immutable = np.ascontiguousarray(layer_capture)
        immutable.setflags(write=False)
        captures.append(immutable)
        geometry_ids.append(tuple(ids))
        coverage_fractions.append(tuple(coverage))
    total = np.asarray(exposure.values - remaining, dtype=np.float64)
    if (
        not np.all(np.isfinite(total))
        or not np.all(np.isfinite(remaining))
        or np.any(total < 0.0)
        or np.any(remaining < 0.0)
        or np.any(total > exposure.values)
    ):
        raise RuntimeError("crystal capture left the physical exposure domain")
    return CrystalCaptureResult(
        captured_exposure=_wrap_exposure(total, exposure),
        remaining_exposure=_wrap_exposure(remaining, exposure),
        layer_captures=tuple(captures),
        geometry_ids=tuple(geometry_ids),
        coverage_fractions=tuple(coverage_fractions),
        profile_id=profile.identity(),
    )


def render_independent_footprint_sum(
    exposure: PhysicalDomainArray, profile: CrystalLightDepletionProfile
) -> np.ndarray:
    """Return the fixed no-depletion control; values may exceed input exposure."""
    _validate_exposure(exposure)
    shape = exposure.values.shape[:2]
    total = np.zeros_like(exposure.values)
    for layer_index in range(profile.layer_count_per_channel):
        for channel_index in range(3):
            geometry = build_crystal_footprint_map(
                shape,
                profile,
                channel_index=channel_index,
                layer_index=layer_index,
            )
            total[..., channel_index] += _footprint_capture(
                exposure.values[..., channel_index],
                geometry,
                profile.capture_efficiency,
            )
    output = np.ascontiguousarray(total, dtype=np.float64)
    output.setflags(write=False)
    return output


__all__ = [
    "CRYSTAL_LIGHT_DEPLETION_SCHEMA",
    "CrystalCaptureResult",
    "CrystalFootprintMap",
    "CrystalLightDepletionError",
    "CrystalLightDepletionProfile",
    "build_crystal_footprint_map",
    "render_crystal_light_depletion",
    "render_independent_footprint_sum",
]
