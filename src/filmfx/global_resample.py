"""Versioned original-coordinate staging for low-frequency film-effect fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fast_blur import gaussian_filter_direct


GLOBAL_RESAMPLE_VERSION = "shape-stable-global-resample-v1"


@dataclass(frozen=True)
class GlobalResamplePlan:
    version: str
    source_shape: tuple[int, int]
    coarse_shape: tuple[int, int]
    spatial_sigma: tuple[float, float]
    coarse_sigma: tuple[float, float]
    scale_factors: tuple[float, float]
    truncate: float


@dataclass(frozen=True)
class StagedGlobalField:
    plan: GlobalResamplePlan
    coarse: np.ndarray

    @property
    def coarse_bytes(self) -> int:
        return int(self.coarse.nbytes)


@dataclass(frozen=True)
class GlobalResampleExecutionMetadata:
    version: str
    source_shape: tuple[int, ...]
    coarse_shape: tuple[int, ...]
    tile_size: int
    tile_count: int
    max_window_shape: tuple[int, ...]
    coarse_bytes: int
    output_bytes: int


def _positive_shape2(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} must be an (H, W) tuple")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{label} dimensions must be positive integers")
    return value


def _spatial_sigma(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("sigma must be a scalar or a two-item tuple")
        result = tuple(float(item) for item in value)
    else:
        result = (float(value), float(value))
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError("sigma must be finite and non-negative")
    return result


def plan_shape_stable_global_resample(
    source_shape: tuple[int, int],
    sigma: float | tuple[float, float],
    *,
    truncate: float = 3.0,
    max_direct_radius: int = 32,
    target_downsampled_sigma: float = 6.0,
) -> GlobalResamplePlan:
    """Freeze one coarse grid from the original field geometry."""

    height, width = _positive_shape2(source_shape, "source_shape")
    spatial_sigma = _spatial_sigma(sigma)
    if not math.isfinite(float(truncate)) or float(truncate) <= 0.0:
        raise ValueError("truncate must be finite and positive")
    if isinstance(max_direct_radius, bool) or not isinstance(max_direct_radius, int) or max_direct_radius < 0:
        raise ValueError("max_direct_radius must be a non-negative integer")
    if not math.isfinite(float(target_downsampled_sigma)) or float(target_downsampled_sigma) <= 0.0:
        raise ValueError("target_downsampled_sigma must be finite and positive")

    max_radius = max(int(round(float(truncate) * item)) for item in spatial_sigma)
    factor = 1
    if max_radius > max_direct_radius and min(height, width) > 2:
        factor = max(1, int(math.floor(max(spatial_sigma) / float(target_downsampled_sigma))))
        factor = min(factor, max(1, min(height, width) // 8))
    if factor <= 1:
        coarse_shape = (height, width)
    else:
        coarse_shape = (max(2, height // factor), max(2, width // factor))
    scale_factors = (height / coarse_shape[0], width / coarse_shape[1])
    coarse_sigma = (
        spatial_sigma[0] / scale_factors[0],
        spatial_sigma[1] / scale_factors[1],
    )
    return GlobalResamplePlan(
        version=GLOBAL_RESAMPLE_VERSION,
        source_shape=(height, width),
        coarse_shape=coarse_shape,
        spatial_sigma=spatial_sigma,
        coarse_sigma=coarse_sigma,
        scale_factors=scale_factors,
        truncate=float(truncate),
    )


def _validate_plan(plan: object) -> GlobalResamplePlan:
    if not isinstance(plan, GlobalResamplePlan) or plan.version != GLOBAL_RESAMPLE_VERSION:
        raise ValueError("plan is not a supported GlobalResamplePlan")
    source_shape = _positive_shape2(plan.source_shape, "plan.source_shape")
    coarse_shape = _positive_shape2(plan.coarse_shape, "plan.coarse_shape")
    if any(coarse > source for coarse, source in zip(coarse_shape, source_shape, strict=True)):
        raise ValueError("plan coarse_shape cannot exceed source_shape")
    spatial_sigma = _spatial_sigma(plan.spatial_sigma)
    coarse_sigma = _spatial_sigma(plan.coarse_sigma)
    if not math.isfinite(float(plan.truncate)) or float(plan.truncate) <= 0.0:
        raise ValueError("plan truncate must be finite and positive")
    expected_scale = tuple(source / coarse for source, coarse in zip(source_shape, coarse_shape, strict=True))
    expected_sigma = tuple(sigma / scale for sigma, scale in zip(spatial_sigma, expected_scale, strict=True))
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(plan.scale_factors, expected_scale, strict=True)
    ):
        raise ValueError("plan scale_factors do not match its shapes")
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(coarse_sigma, expected_sigma, strict=True)
    ):
        raise ValueError("plan coarse_sigma does not match its geometry")
    return plan


def _area_downsample_axis(array: np.ndarray, destination: int, axis: int) -> np.ndarray:
    source = array.shape[axis]
    if source == destination:
        return array.astype(np.float32, copy=True)
    moved = np.moveaxis(array, axis, 0)
    output = np.empty((destination,) + moved.shape[1:], dtype=np.float32)
    scale = source / destination
    for index in range(destination):
        start = index * scale
        end = (index + 1) * scale
        first = int(math.floor(start))
        stop = int(math.ceil(end))
        cells = np.arange(first, stop, dtype=np.float64)
        weights = np.minimum(end, cells + 1.0) - np.maximum(start, cells)
        weights = (weights / (end - start)).astype(np.float32)
        output[index] = np.tensordot(weights, moved[first:stop], axes=(0, 0))
    return np.moveaxis(output, 0, axis).astype(np.float32, copy=False)


def _area_downsample(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    # Reduce width first so the transient height pass is already narrow.
    horizontal = _area_downsample_axis(array, shape[1], axis=1)
    return _area_downsample_axis(horizontal, shape[0], axis=0)


def _validate_field(array: object, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    if not isinstance(array, np.ndarray) or array.ndim not in {2, 3}:
        raise ValueError("field must be a 2-D or HWC numpy array")
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("field must be finite float32")
    if any(size <= 0 for size in array.shape):
        raise ValueError("field dimensions must be non-zero")
    if shape is not None and array.shape[:2] != shape:
        raise ValueError(f"field shape {array.shape[:2]} does not match plan {shape}")
    return array


def build_shape_stable_global_stage(array: np.ndarray, plan: GlobalResamplePlan) -> StagedGlobalField:
    """Area-stage and direct-blur one immutable coarse field."""

    plan = _validate_plan(plan)
    field = _validate_field(array, shape=plan.source_shape)
    coarse = _area_downsample(field, plan.coarse_shape)
    sigma = plan.coarse_sigma + ((0.0,) if field.ndim == 3 else ())
    coarse = gaussian_filter_direct(coarse, sigma=sigma, truncate=plan.truncate)
    coarse = np.ascontiguousarray(coarse, dtype=np.float32)
    coarse.setflags(write=False)
    return StagedGlobalField(plan=plan, coarse=coarse)


def _validate_stage(stage: object) -> StagedGlobalField:
    if not isinstance(stage, StagedGlobalField):
        raise ValueError("stage must be a StagedGlobalField")
    _validate_plan(stage.plan)
    coarse = _validate_field(stage.coarse, shape=stage.plan.coarse_shape)
    if coarse.flags.writeable:
        raise ValueError("staged coarse field must be read-only")
    return stage


def reconstruct_shape_stable_window(
    stage: StagedGlobalField,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Reconstruct one window using only original-image coordinates."""

    stage = _validate_stage(stage)
    height, width = stage.plan.source_shape
    bounds = (y0, y1, x0, x1)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in bounds):
        raise ValueError("window bounds must be integers")
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("window bounds are outside the source field")

    coarse = stage.coarse[..., None] if stage.coarse.ndim == 2 else stage.coarse
    coarse_height, coarse_width = stage.plan.coarse_shape
    ys = (np.arange(y0, y1, dtype=np.float64) + 0.5) * coarse_height / height - 0.5
    xs = (np.arange(x0, x1, dtype=np.float64) + 0.5) * coarse_width / width - 0.5
    y_floor = np.floor(ys).astype(np.int64)
    x_floor = np.floor(xs).astype(np.int64)
    wy = (ys - y_floor).astype(np.float32)
    wx = (xs - x_floor).astype(np.float32)
    y_low = np.clip(y_floor, 0, coarse_height - 1)
    y_high = np.clip(y_floor + 1, 0, coarse_height - 1)
    x_low = np.clip(x_floor, 0, coarse_width - 1)
    x_high = np.clip(x_floor + 1, 0, coarse_width - 1)

    output = np.empty((y1 - y0, x1 - x0, coarse.shape[2]), dtype=np.float32)
    left_weight = (1.0 - wx)[:, None]
    right_weight = wx[:, None]
    for row in range(output.shape[0]):
        top = coarse[y_low[row], x_low] * left_weight + coarse[y_low[row], x_high] * right_weight
        bottom = coarse[y_high[row], x_low] * left_weight + coarse[y_high[row], x_high] * right_weight
        output[row] = top * np.float32(1.0 - wy[row]) + bottom * wy[row]
    return output[..., 0] if stage.coarse.ndim == 2 else output


def reconstruct_shape_stable_full(stage: StagedGlobalField) -> np.ndarray:
    height, width = _validate_stage(stage).plan.source_shape
    return reconstruct_shape_stable_window(stage, y0=0, y1=height, x0=0, x1=width)


def reconstruct_shape_stable_tiled(
    stage: StagedGlobalField,
    *,
    tile_size: int,
) -> tuple[np.ndarray, GlobalResampleExecutionMetadata]:
    """Assemble the staged field in row-major original-coordinate windows."""

    stage = _validate_stage(stage)
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    height, width = stage.plan.source_shape
    channels = stage.coarse.shape[2:] if stage.coarse.ndim == 3 else ()
    output = np.empty((height, width) + channels, dtype=np.float32)
    tile_count = 0
    max_height = 0
    max_width = 0
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            output[y0:y1, x0:x1] = reconstruct_shape_stable_window(
                stage, y0=y0, y1=y1, x0=x0, x1=x1
            )
            tile_count += 1
            max_height = max(max_height, y1 - y0)
            max_width = max(max_width, x1 - x0)
    return output, GlobalResampleExecutionMetadata(
        version=stage.plan.version,
        source_shape=tuple(int(item) for item in output.shape),
        coarse_shape=tuple(int(item) for item in stage.coarse.shape),
        tile_size=tile_size,
        tile_count=tile_count,
        max_window_shape=(max_height, max_width) + channels,
        coarse_bytes=stage.coarse_bytes,
        output_bytes=int(output.nbytes),
    )
