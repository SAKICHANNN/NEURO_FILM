"""Piecewise analytic marked-cloud integration through a circular aperture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby, pairwise

import numpy as np

from .developed_structure import DevelopedStructureContext


class AnalyticCloudApertureError(ValueError):
    """Raised when analytic cloud/aperture inputs violate the contract."""


@dataclass(frozen=True)
class AnalyticApertureResult:
    values: np.ndarray


@dataclass(frozen=True)
class _Boundary:
    center_y: float
    center_x: float
    radius: float
    sign: int

    def value(self, x_coordinate: float) -> float:
        delta = x_coordinate - self.center_x
        chord = math.sqrt(max(self.radius * self.radius - delta * delta, 0.0))
        return self.center_y + self.sign * chord

    def primitive(self, x_coordinate: float) -> float:
        delta = x_coordinate - self.center_x
        clipped = min(max(delta / self.radius, -1.0), 1.0)
        chord = math.sqrt(max(self.radius * self.radius - delta * delta, 0.0))
        arc = 0.5 * (delta * chord + self.radius * self.radius * math.asin(clipped))
        return self.center_y * x_coordinate + self.sign * arc


def _circle_intersection_x(
    first_center: np.ndarray,
    first_radius: float,
    second_center: np.ndarray,
    second_radius: float,
) -> tuple[float, ...]:
    delta = second_center - first_center
    distance = float(np.hypot(delta[0], delta[1]))
    if (
        distance == 0.0
        or distance >= first_radius + second_radius
        or distance <= abs(first_radius - second_radius)
    ):
        return ()
    along = (
        first_radius * first_radius
        - second_radius * second_radius
        + distance * distance
    ) / (2.0 * distance)
    height_squared = first_radius * first_radius - along * along
    if height_squared <= 0.0:
        return ()
    midpoint = first_center + (along / distance) * delta
    x_offset = math.sqrt(height_squared) * delta[0] / distance
    return (float(midpoint[1] - x_offset), float(midpoint[1] + x_offset))


def _integral_between(boundary: _Boundary, start: float, stop: float) -> float:
    return boundary.primitive(stop) - boundary.primitive(start)


def integrate_marked_cloud_aperture_analytic(
    cloud_centers_yx: np.ndarray,
    cloud_radii: np.ndarray | float,
    *,
    mark_optical_density: float,
    aperture_center_yx: tuple[float, float],
    aperture_radius: float,
    integration_axis: str = "x",
) -> float:
    """Evaluate the exact piecewise circle-chord integral in float64."""

    centers = np.asarray(cloud_centers_yx, dtype=np.float64)
    if centers.size == 0:
        centers = np.empty((0, 2), dtype=np.float64)
    radii = np.asarray(cloud_radii, dtype=np.float64)
    if radii.ndim == 0:
        radii = np.full(centers.shape[0], float(radii), dtype=np.float64)
    aperture_center = np.asarray(aperture_center_yx, dtype=np.float64)
    if (
        centers.ndim != 2
        or centers.shape[1:] != (2,)
        or radii.shape != (centers.shape[0],)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
        or aperture_center.shape != (2,)
        or not np.all(np.isfinite(aperture_center))
        or not math.isfinite(mark_optical_density)
        or mark_optical_density <= 0.0
        or not math.isfinite(aperture_radius)
        or aperture_radius <= 0.0
        or integration_axis not in {"x", "y"}
    ):
        raise AnalyticCloudApertureError("invalid analytic aperture request")
    if integration_axis == "y":
        centers = centers[:, ::-1]
        aperture_center = aperture_center[::-1]

    distance = np.linalg.norm(centers - aperture_center[None, :], axis=1)
    selected = distance < (radii + aperture_radius)
    centers = centers[selected]
    radii = radii[selected]
    center_y, center_x = (float(value) for value in aperture_center)
    lower_x = center_x - aperture_radius
    upper_x = center_x + aperture_radius
    circles = [(aperture_center, aperture_radius), *zip(centers, radii, strict=True)]
    breakpoints = {lower_x, upper_x}
    for cloud_center, radius in zip(centers, radii, strict=True):
        breakpoints.add(float(cloud_center[1] - radius))
        breakpoints.add(float(cloud_center[1] + radius))
    for first in range(len(circles)):
        for second in range(first + 1, len(circles)):
            breakpoints.update(
                _circle_intersection_x(
                    np.asarray(circles[first][0], dtype=np.float64),
                    float(circles[first][1]),
                    np.asarray(circles[second][0], dtype=np.float64),
                    float(circles[second][1]),
                )
            )
    ordered_x = sorted(value for value in breakpoints if lower_x <= value <= upper_x)
    aperture_lower = _Boundary(center_y, center_x, aperture_radius, -1)
    aperture_upper = _Boundary(center_y, center_x, aperture_radius, 1)
    integral = 0.0

    for start_x, stop_x in pairwise(ordered_x):
        if not start_x < stop_x:
            continue
        midpoint_x = 0.5 * (start_x + stop_x)
        lower_aperture_y = aperture_lower.value(midpoint_x)
        upper_aperture_y = aperture_upper.value(midpoint_x)
        events: list[tuple[float, _Boundary, int]] = []
        coverage = 0
        for cloud_center, radius in zip(centers, radii, strict=True):
            if abs(midpoint_x - cloud_center[1]) >= radius:
                continue
            lower = _Boundary(
                float(cloud_center[0]), float(cloud_center[1]), float(radius), -1
            )
            upper = _Boundary(
                float(cloud_center[0]), float(cloud_center[1]), float(radius), 1
            )
            lower_y = lower.value(midpoint_x)
            upper_y = upper.value(midpoint_x)
            if lower_y < lower_aperture_y < upper_y:
                coverage += 1
            if lower_aperture_y < lower_y < upper_aperture_y:
                events.append((lower_y, lower, 1))
            if lower_aperture_y < upper_y < upper_aperture_y:
                events.append((upper_y, upper, -1))

        current = aperture_lower
        for _, grouped in groupby(
            sorted(events, key=lambda item: item[0]), key=lambda item: item[0]
        ):
            group = list(grouped)
            boundary = group[0][1]
            transmission = 10.0 ** (-mark_optical_density * coverage)
            integral += transmission * (
                _integral_between(boundary, start_x, stop_x)
                - _integral_between(current, start_x, stop_x)
            )
            coverage += sum(item[2] for item in group)
            current = boundary
        transmission = 10.0 ** (-mark_optical_density * coverage)
        integral += transmission * (
            _integral_between(aperture_upper, start_x, stop_x)
            - _integral_between(current, start_x, stop_x)
        )

    normalized = integral / (math.pi * aperture_radius * aperture_radius)
    if not math.isfinite(normalized) or normalized <= 0.0 or normalized > 1.0 + 1e-11:
        raise AnalyticCloudApertureError(
            "analytic aperture result left transmittance domain"
        )
    return min(normalized, 1.0)


def render_analytic_cloud_aperture(
    context: DevelopedStructureContext,
    *,
    aperture_diameter_um: float,
    integration_axis: str = "x",
    row_partition: int | None = None,
) -> AnalyticApertureResult:
    if context.material != "colour-dye-cloud" or context.output_zoom != 1:
        raise AnalyticCloudApertureError(
            "analytic integration requires target-pitch colour cloud geometry"
        )
    if not math.isfinite(aperture_diameter_um) or aperture_diameter_um <= 0.0:
        raise AnalyticCloudApertureError("aperture diameter must be positive")
    if row_partition is not None and (
        not isinstance(row_partition, int)
        or isinstance(row_partition, bool)
        or row_partition < 1
    ):
        raise AnalyticCloudApertureError("row_partition must be positive")
    aperture_radius = aperture_diameter_um / (2.0 * context.output_pixel_pitch_um)
    values = np.empty((*context.input_shape, 3), dtype=np.float64)
    step = context.input_shape[0] if row_partition is None else row_partition
    for row_start in range(0, context.input_shape[0], step):
        row_stop = min(row_start + step, context.input_shape[0])
        for row in range(row_start, row_stop):
            for column in range(context.input_shape[1]):
                for layer, (centers, radius, mark) in enumerate(
                    zip(
                        context.centers_by_layer,
                        context.radius_input_pixels,
                        context.mark_optical_density,
                        strict=True,
                    )
                ):
                    values[row, column, layer] = (
                        integrate_marked_cloud_aperture_analytic(
                            centers,
                            radius,
                            mark_optical_density=mark,
                            aperture_center_yx=(row + 0.5, column + 0.5),
                            aperture_radius=aperture_radius,
                            integration_axis=integration_axis,
                        )
                    )
    values.setflags(write=False)
    return AnalyticApertureResult(values)


__all__ = [
    "AnalyticApertureResult",
    "AnalyticCloudApertureError",
    "integrate_marked_cloud_aperture_analytic",
    "render_analytic_cloud_aperture",
]
