"""Clean-room deterministic Boolean/Poisson film-grain representation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


BOOLEAN_GRAIN_SCHEMA = "filmfx.boolean_grain.v1"


def _readonly_float64(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _validate_input_intensity(
    intensity: np.ndarray,
    *,
    maximum_input_intensity: float,
) -> np.ndarray:
    values = np.asarray(intensity, dtype=np.float64)
    if (
        values.ndim != 2
        or not values.size
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > maximum_input_intensity)
    ):
        raise ValueError(
            "intensity must be a non-empty finite 2D array within the configured range"
        )
    return values


@dataclass(frozen=True)
class BooleanGrainContext:
    """Serializable random-disk context independent of output partitioning."""

    input_intensity: np.ndarray
    grain_centers_yx: np.ndarray
    grain_radii: np.ndarray
    monte_carlo_offsets_yx: np.ndarray
    radius_input_pixels: float
    gaussian_filter_sigma_output_pixels: float
    maximum_input_intensity: float
    epsilon: float
    seed: int

    def __post_init__(self) -> None:
        intensity = _validate_input_intensity(
            self.input_intensity,
            maximum_input_intensity=self.maximum_input_intensity,
        )
        centers = np.asarray(self.grain_centers_yx, dtype=np.float64)
        radii = np.asarray(self.grain_radii, dtype=np.float64)
        offsets = np.asarray(self.monte_carlo_offsets_yx, dtype=np.float64)
        if (
            centers.ndim != 2
            or centers.shape[1:] != (2,)
            or not np.all(np.isfinite(centers))
            or np.any(centers[:, 0] < 0.0)
            or np.any(centers[:, 0] >= intensity.shape[0])
            or np.any(centers[:, 1] < 0.0)
            or np.any(centers[:, 1] >= intensity.shape[1])
        ):
            raise ValueError("grain centers must be finite input-domain y/x rows")
        if (
            radii.shape != (len(centers),)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError("grain radii must be positive and match centers")
        if (
            offsets.ndim != 2
            or offsets.shape[1:] != (2,)
            or len(offsets) == 0
            or not np.all(np.isfinite(offsets))
        ):
            raise ValueError("Monte Carlo offsets must be finite y/x rows")
        positive = (
            self.radius_input_pixels,
            self.gaussian_filter_sigma_output_pixels,
            self.maximum_input_intensity,
            self.epsilon,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("context scalar parameters must be finite and positive")
        if self.maximum_input_intensity >= 1.0:
            raise ValueError("maximum_input_intensity must be below one")
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "input_intensity", _readonly_float64(intensity))
        object.__setattr__(self, "grain_centers_yx", _readonly_float64(centers))
        object.__setattr__(self, "grain_radii", _readonly_float64(radii))
        object.__setattr__(
            self,
            "monte_carlo_offsets_yx",
            _readonly_float64(offsets),
        )

    @property
    def input_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.input_intensity.shape)

    @property
    def grain_count(self) -> int:
        return int(len(self.grain_radii))

    @property
    def monte_carlo_samples(self) -> int:
        return int(len(self.monte_carlo_offsets_yx))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BOOLEAN_GRAIN_SCHEMA,
            "input_intensity": self.input_intensity.tolist(),
            "grain_centers_yx": self.grain_centers_yx.tolist(),
            "grain_radii": self.grain_radii.tolist(),
            "monte_carlo_offsets_yx": self.monte_carlo_offsets_yx.tolist(),
            "radius_input_pixels": self.radius_input_pixels,
            "gaussian_filter_sigma_output_pixels": (
                self.gaussian_filter_sigma_output_pixels
            ),
            "maximum_input_intensity": self.maximum_input_intensity,
            "epsilon": self.epsilon,
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BooleanGrainContext":
        if payload.get("schema") != BOOLEAN_GRAIN_SCHEMA:
            raise ValueError("unsupported Boolean grain schema")
        return cls(
            input_intensity=np.asarray(payload["input_intensity"], dtype=np.float64),
            grain_centers_yx=np.asarray(
                payload["grain_centers_yx"], dtype=np.float64
            ),
            grain_radii=np.asarray(payload["grain_radii"], dtype=np.float64),
            monte_carlo_offsets_yx=np.asarray(
                payload["monte_carlo_offsets_yx"], dtype=np.float64
            ),
            radius_input_pixels=float(payload["radius_input_pixels"]),
            gaussian_filter_sigma_output_pixels=float(
                payload["gaussian_filter_sigma_output_pixels"]
            ),
            maximum_input_intensity=float(payload["maximum_input_intensity"]),
            epsilon=float(payload["epsilon"]),
            seed=int(payload["seed"]),
        )

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_boolean_grain_context(
    intensity: np.ndarray,
    *,
    radius_input_pixels: float,
    monte_carlo_samples: int,
    gaussian_filter_sigma_output_pixels: float,
    maximum_input_intensity: float,
    epsilon: float,
    seed: int,
) -> BooleanGrainContext:
    """Sample an inhomogeneous constant-radius Poisson disk field."""

    values = _validate_input_intensity(
        intensity,
        maximum_input_intensity=maximum_input_intensity,
    )
    positive = (
        radius_input_pixels,
        gaussian_filter_sigma_output_pixels,
        maximum_input_intensity,
        epsilon,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("context settings must be finite and positive")
    if maximum_input_intensity >= 1.0:
        raise ValueError("maximum_input_intensity must be below one")
    if monte_carlo_samples <= 0 or seed < 0:
        raise ValueError("samples and seed are invalid")

    rng = np.random.default_rng(int(seed))
    offsets = rng.normal(
        loc=0.0,
        scale=float(gaussian_filter_sigma_output_pixels),
        size=(int(monte_carlo_samples), 2),
    )
    expected_area = math.pi * radius_input_pixels * radius_input_pixels
    poisson_intensity = -np.log1p(-values) / (expected_area + epsilon)
    counts = rng.poisson(poisson_intensity)
    total = int(np.sum(counts, dtype=np.int64))
    centers = np.empty((total, 2), dtype=np.float64)
    cursor = 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            count = int(counts[row, column])
            if count:
                centers[cursor : cursor + count, 0] = row + rng.random(count)
                centers[cursor : cursor + count, 1] = column + rng.random(count)
                cursor += count
    if cursor != total:
        raise RuntimeError("grain context construction count mismatch")
    radii = np.full(total, radius_input_pixels, dtype=np.float64)
    return BooleanGrainContext(
        input_intensity=values,
        grain_centers_yx=centers,
        grain_radii=radii,
        monte_carlo_offsets_yx=offsets,
        radius_input_pixels=radius_input_pixels,
        gaussian_filter_sigma_output_pixels=gaussian_filter_sigma_output_pixels,
        maximum_input_intensity=maximum_input_intensity,
        epsilon=epsilon,
        seed=int(seed),
    )


def render_boolean_grain_region(
    context: BooleanGrainContext,
    *,
    output_zoom: int,
    output_origin_yx: tuple[int, int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize one output region using global output coordinates."""

    if not isinstance(output_zoom, int) or output_zoom <= 0:
        raise ValueError("output_zoom must be a positive integer")
    if len(output_origin_yx) != 2 or len(output_shape) != 2:
        raise ValueError("origin and shape must contain y/x")
    origin_y, origin_x = (int(value) for value in output_origin_yx)
    height, width = (int(value) for value in output_shape)
    full_height = context.input_shape[0] * output_zoom
    full_width = context.input_shape[1] * output_zoom
    if (
        origin_y < 0
        or origin_x < 0
        or height <= 0
        or width <= 0
        or origin_y + height > full_height
        or origin_x + width > full_width
    ):
        raise ValueError("output region lies outside the zoomed image")

    accumulated = np.zeros((height, width), dtype=np.uint32)
    for offset_y, offset_x in context.monte_carlo_offsets_yx:
        covered = np.zeros((height, width), dtype=bool)
        for (center_y, center_x), radius in zip(
            context.grain_centers_yx,
            context.grain_radii,
            strict=True,
        ):
            output_center_y = output_zoom * center_y + offset_y
            output_center_x = output_zoom * center_x + offset_x
            output_radius = output_zoom * radius
            local_y0 = max(
                0,
                int(math.ceil(output_center_y - output_radius - origin_y)),
            )
            local_y1 = min(
                height - 1,
                int(math.floor(output_center_y + output_radius - origin_y)),
            )
            local_x0 = max(
                0,
                int(math.ceil(output_center_x - output_radius - origin_x)),
            )
            local_x1 = min(
                width - 1,
                int(math.floor(output_center_x + output_radius - origin_x)),
            )
            if local_y0 > local_y1 or local_x0 > local_x1:
                continue
            rows = np.arange(local_y0, local_y1 + 1, dtype=np.float64) + origin_y
            columns = (
                np.arange(local_x0, local_x1 + 1, dtype=np.float64) + origin_x
            )
            squared = (
                (rows[:, None] - output_center_y) ** 2
                + (columns[None, :] - output_center_x) ** 2
            )
            covered[
                local_y0 : local_y1 + 1,
                local_x0 : local_x1 + 1,
            ] |= squared < output_radius * output_radius
        accumulated += covered
    output = accumulated.astype(np.float32) / np.float32(
        context.monte_carlo_samples
    )
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("Boolean grain output escaped [0,1]")
    output.setflags(write=False)
    return output


def render_boolean_grain(
    context: BooleanGrainContext,
    *,
    output_zoom: int,
) -> np.ndarray:
    return render_boolean_grain_region(
        context,
        output_zoom=output_zoom,
        output_origin_yx=(0, 0),
        output_shape=(
            context.input_shape[0] * output_zoom,
            context.input_shape[1] * output_zoom,
        ),
    )


__all__ = [
    "BOOLEAN_GRAIN_SCHEMA",
    "BooleanGrainContext",
    "build_boolean_grain_context",
    "render_boolean_grain",
    "render_boolean_grain_region",
]
