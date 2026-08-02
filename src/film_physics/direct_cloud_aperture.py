"""Direct circular-aperture quadrature over continuous marked dye clouds."""

from __future__ import annotations

import math

import numpy as np

from .developed_structure import DevelopedStructureContext


class DirectCloudApertureError(ValueError):
    """Raised when direct cloud/aperture inputs violate the reference contract."""


def concentric_golden_angle_disk_points(sample_count: int) -> np.ndarray:
    """Return deterministic equal-mass midpoint points in the unit disk."""

    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 1
    ):
        raise DirectCloudApertureError("sample_count must be a positive integer")
    index = np.arange(sample_count, dtype=np.float64)
    radius = np.sqrt((index + 0.5) / sample_count)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angle = index * golden_angle
    points = np.stack((radius * np.sin(angle), radius * np.cos(angle)), axis=1)
    points.setflags(write=False)
    return points


def render_direct_cloud_aperture_region(
    context: DevelopedStructureContext,
    *,
    aperture_diameter_um: float,
    sample_count: int,
    point_chunk_size: int,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Average per-point dye-cloud transmittance inside each target aperture."""

    if context.material != "colour-dye-cloud" or context.output_zoom != 1:
        raise DirectCloudApertureError(
            "direct quadrature requires a target-pitch colour dye-cloud context"
        )
    if (
        not math.isfinite(aperture_diameter_um)
        or aperture_diameter_um <= 0.0
        or not isinstance(point_chunk_size, int)
        or isinstance(point_chunk_size, bool)
        or point_chunk_size < 1
        or len(origin_yx) != 2
        or len(shape) != 2
        or any(not isinstance(value, int) for value in (*origin_yx, *shape))
    ):
        raise DirectCloudApertureError("invalid direct quadrature request")
    y0, x0 = origin_yx
    height, width = shape
    if (
        y0 < 0
        or x0 < 0
        or height < 1
        or width < 1
        or y0 + height > context.input_shape[0]
        or x0 + width > context.input_shape[1]
    ):
        raise DirectCloudApertureError("direct quadrature region is out of bounds")

    disk = concentric_golden_angle_disk_points(sample_count)
    aperture_radius_pixels = aperture_diameter_um / (
        2.0 * context.output_pixel_pitch_um
    )
    ys, xs = np.meshgrid(
        np.arange(y0, y0 + height, dtype=np.float64) + 0.5,
        np.arange(x0, x0 + width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    centres = np.stack((ys, xs), axis=-1)
    points = (
        centres[:, :, None, :] + disk[None, None, :, :] * aperture_radius_pixels
    ).reshape(-1, 2)
    output = np.empty((height * width, sample_count, 3), dtype=np.float64)
    for layer, (clouds, radius, mark) in enumerate(
        zip(
            context.centers_by_layer,
            context.radius_input_pixels,
            context.mark_optical_density,
            strict=True,
        )
    ):
        transmission = np.empty(points.shape[0], dtype=np.float64)
        radius_squared = radius * radius
        for start in range(0, points.shape[0], point_chunk_size):
            stop = min(start + point_chunk_size, points.shape[0])
            chunk = points[start:stop]
            density = np.zeros(stop - start, dtype=np.float64)
            for cloud_start in range(0, len(clouds), point_chunk_size):
                cloud_stop = min(cloud_start + point_chunk_size, len(clouds))
                cloud_chunk = clouds[cloud_start:cloud_stop]
                squared_distance = np.sum(
                    np.square(chunk[:, None, :] - cloud_chunk[None, :, :]),
                    axis=2,
                    dtype=np.float64,
                )
                density += mark * np.count_nonzero(
                    squared_distance < radius_squared, axis=1
                )
            transmission[start:stop] = np.power(10.0, -density)
        output[:, :, layer] = transmission.reshape(height * width, sample_count)
    averaged = output.mean(axis=1, dtype=np.float64).reshape(height, width, 3)
    if (
        not np.all(np.isfinite(averaged))
        or np.any(averaged <= 0.0)
        or np.any(averaged > 1.0)
    ):
        raise RuntimeError("direct cloud aperture left transmittance domain")
    return averaged


def render_direct_cloud_aperture(
    context: DevelopedStructureContext,
    *,
    aperture_diameter_um: float,
    sample_count: int,
    point_chunk_size: int,
    row_partition: int | None = None,
) -> np.ndarray:
    if row_partition is not None and (
        not isinstance(row_partition, int)
        or isinstance(row_partition, bool)
        or row_partition < 1
    ):
        raise DirectCloudApertureError("row_partition must be positive")
    step = context.input_shape[0] if row_partition is None else row_partition
    output = np.empty((*context.input_shape, 3), dtype=np.float64)
    for y0 in range(0, context.input_shape[0], step):
        height = min(step, context.input_shape[0] - y0)
        output[y0 : y0 + height] = render_direct_cloud_aperture_region(
            context,
            aperture_diameter_um=aperture_diameter_um,
            sample_count=sample_count,
            point_chunk_size=point_chunk_size,
            origin_yx=(y0, 0),
            shape=(height, context.input_shape[1]),
        )
    return output


__all__ = [
    "DirectCloudApertureError",
    "concentric_golden_angle_disk_points",
    "render_direct_cloud_aperture",
    "render_direct_cloud_aperture_region",
]
