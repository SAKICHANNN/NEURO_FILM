"""Bounded monotone triangular RGB transport with analytic inverse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


PARAMETER_COUNT = 14


class TriangularLogitTransportError(ValueError):
    """Raised when the explicit transport contract is violated."""


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    if (
        value.ndim < 1
        or value.shape[-1] != 3
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or np.any(value > 1.0)
    ):
        raise TriangularLogitTransportError(
            "RGB must be finite, cube-bounded and end in three channels"
        )
    return value


def _validate_parameters(parameters: np.ndarray) -> np.ndarray:
    value = np.asarray(parameters, dtype=np.float64)
    if value.shape != (PARAMETER_COUNT,) or not np.all(np.isfinite(value)):
        raise TriangularLogitTransportError(
            "transport requires fourteen finite parameters"
        )
    return value


def _logit(value: np.ndarray) -> np.ndarray:
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _forward_channel(
    source: np.ndarray, log_scale: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    result = source.copy()
    interior = (source > 0.0) & (source < 1.0)
    if np.any(interior):
        result[interior] = _sigmoid(
            np.exp(log_scale[interior]) * _logit(source[interior])
            + shift[interior]
        )
    return result


def _inverse_channel(
    target: np.ndarray, log_scale: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    result = target.copy()
    interior = (target > 0.0) & (target < 1.0)
    if np.any(interior):
        result[interior] = _sigmoid(
            (_logit(target[interior]) - shift[interior])
            / np.exp(log_scale[interior])
        )
    return result


def _green_basis(red_output: np.ndarray) -> np.ndarray:
    centered = red_output - 0.5
    return np.stack((np.ones_like(centered), centered), axis=-1)


def _blue_basis(
    red_output: np.ndarray, green_output: np.ndarray
) -> np.ndarray:
    red = red_output - 0.5
    green = green_output - 0.5
    return np.stack(
        (np.ones_like(red), red, green, red * green), axis=-1
    )


@dataclass(frozen=True)
class TriangularLogitTransport:
    """Explicit cube-preserving RGB transform with a triangular Jacobian."""

    parameters: np.ndarray
    dose: float = 1.0

    def __post_init__(self) -> None:
        parameters = _validate_parameters(self.parameters).copy()
        dose = float(self.dose)
        if not np.isfinite(dose) or dose < 0.0 or dose > 1.0:
            raise TriangularLogitTransportError("dose must be within [0, 1]")
        parameters.setflags(write=False)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "dose", dose)

    @property
    def effective_parameters(self) -> np.ndarray:
        return self.parameters * self.dose

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        source = _validate_rgb(rgb)
        p = self.effective_parameters
        red_log_scale = np.full(source.shape[:-1], p[0])
        red_shift = np.full(source.shape[:-1], p[1])
        red = _forward_channel(source[..., 0], red_log_scale, red_shift)
        green_features = _green_basis(red)
        green = _forward_channel(
            source[..., 1],
            green_features @ p[2:4],
            green_features @ p[4:6],
        )
        blue_features = _blue_basis(red, green)
        blue = _forward_channel(
            source[..., 2],
            blue_features @ p[6:10],
            blue_features @ p[10:14],
        )
        output = np.stack((red, green, blue), axis=-1)
        if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(
            output > 1.0
        ):
            raise TriangularLogitTransportError("transport escaped RGB cube")
        return output

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        target = _validate_rgb(rgb)
        p = self.effective_parameters
        red_log_scale = np.full(target.shape[:-1], p[0])
        red_shift = np.full(target.shape[:-1], p[1])
        red = _inverse_channel(target[..., 0], red_log_scale, red_shift)
        green_features = _green_basis(target[..., 0])
        green = _inverse_channel(
            target[..., 1],
            green_features @ p[2:4],
            green_features @ p[4:6],
        )
        blue_features = _blue_basis(target[..., 0], target[..., 1])
        blue = _inverse_channel(
            target[..., 2],
            blue_features @ p[6:10],
            blue_features @ p[10:14],
        )
        output = np.stack((red, green, blue), axis=-1)
        if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(
            output > 1.0
        ):
            raise TriangularLogitTransportError("inverse escaped RGB cube")
        return output


def fit_triangular_logit_transport(
    source: np.ndarray,
    target: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    sample_stride: int,
    identity_shrinkage: float,
    maximum_evaluations: int,
) -> tuple[np.ndarray, bool]:
    """Fit the three triangular stages sequentially on paired RGB samples."""

    source_rgb = _validate_rgb(source)
    target_rgb = _validate_rgb(target)
    if source_rgb.shape != target_rgb.shape:
        raise TriangularLogitTransportError("source/target shape mismatch")
    lower = _validate_parameters(lower_bounds)
    upper = _validate_parameters(upper_bounds)
    if np.any(lower >= upper):
        raise TriangularLogitTransportError("invalid parameter bounds")
    stride = int(sample_stride)
    if stride < 1 or identity_shrinkage < 0.0 or maximum_evaluations < 1:
        raise TriangularLogitTransportError("invalid fit settings")
    sampled_source = source_rgb[::stride, ::stride].reshape(-1, 3)
    sampled_target = target_rgb[::stride, ::stride].reshape(-1, 3)
    parameters = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    successes = []

    def fit_stage(
        indices: slice,
        channel: int,
        features: np.ndarray,
    ) -> None:
        source_channel = sampled_source[:, channel]
        target_channel = sampled_target[:, channel]
        interior = (
            (source_channel > 0.0)
            & (source_channel < 1.0)
            & (target_channel > 0.0)
            & (target_channel < 1.0)
        )
        if np.count_nonzero(interior) < features.shape[1] * 2:
            successes.append(False)
            return
        x = _logit(source_channel[interior])
        y = _logit(target_channel[interior])
        design = features[interior]
        width = design.shape[1]
        span = np.maximum(upper[indices] - lower[indices], 1.0e-12)

        def residual(stage_parameters: np.ndarray) -> np.ndarray:
            prediction = (
                np.exp(design @ stage_parameters[:width]) * x
                + design @ stage_parameters[width:]
            )
            regularizer = (
                np.sqrt(identity_shrinkage) * stage_parameters / span
            )
            return np.concatenate((prediction - y, regularizer))

        result = least_squares(
            residual,
            x0=np.zeros(width * 2, dtype=np.float64),
            bounds=(lower[indices], upper[indices]),
            method="trf",
            max_nfev=maximum_evaluations,
        )
        parameters[indices] = result.x
        successes.append(bool(result.success))

    fit_stage(slice(0, 2), 0, np.ones((len(sampled_source), 1)))
    red = TriangularLogitTransport(parameters).apply(sampled_source)[..., 0]
    fit_stage(slice(2, 6), 1, _green_basis(red))
    red_green = TriangularLogitTransport(parameters).apply(sampled_source)
    fit_stage(
        slice(6, 14),
        2,
        _blue_basis(red_green[..., 0], red_green[..., 1]),
    )
    return parameters, all(successes)


def transport_diagnostics(
    transport: TriangularLogitTransport,
    *,
    grid_size: int,
    finite_difference: float,
) -> dict[str, float | int]:
    """Measure interior Jacobian and inverse accuracy on a fixed cube grid."""

    if grid_size < 3 or finite_difference <= 0.0:
        raise TriangularLogitTransportError("invalid diagnostic grid")
    axis = np.linspace(0.02, 0.98, grid_size, dtype=np.float64)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    flat = points.reshape(-1, 3)
    jacobian = np.empty((len(flat), 3, 3), dtype=np.float64)
    for channel in range(3):
        plus = flat.copy()
        minus = flat.copy()
        plus[:, channel] += finite_difference
        minus[:, channel] -= finite_difference
        jacobian[:, :, channel] = (
            transport.apply(plus) - transport.apply(minus)
        ) / (2.0 * finite_difference)
    determinants = np.linalg.det(jacobian)
    conditions = np.asarray(
        [np.linalg.cond(matrix) for matrix in jacobian], dtype=np.float64
    )
    output = transport.apply(flat)
    restored = transport.inverse(output)
    return {
        "minimum_determinant": float(np.min(determinants)),
        "nonpositive_determinant_count": int(
            np.count_nonzero(determinants <= 0.0)
        ),
        "maximum_condition": float(np.max(conditions)),
        "maximum_inverse_roundtrip_error": float(
            np.max(np.abs(restored - flat))
        ),
    }


def select_safe_transport(
    *,
    parameters: np.ndarray,
    grid_size: int,
    finite_difference: float,
    minimum_jacobian_determinant: float,
    maximum_jacobian_condition: float,
    bisection_iterations: int,
) -> tuple[TriangularLogitTransport, dict[str, float | int]]:
    """Shrink parameters toward identity until frozen Jacobian gates pass."""

    params = _validate_parameters(parameters)

    def inspect(dose: float) -> tuple[bool, dict[str, float | int]]:
        candidate = TriangularLogitTransport(params, dose=dose)
        diagnostics = transport_diagnostics(
            candidate,
            grid_size=grid_size,
            finite_difference=finite_difference,
        )
        passed = (
            diagnostics["nonpositive_determinant_count"] == 0
            and diagnostics["minimum_determinant"]
            >= minimum_jacobian_determinant
            and diagnostics["maximum_condition"]
            <= maximum_jacobian_condition
        )
        return passed, diagnostics

    passed, diagnostics = inspect(1.0)
    if passed:
        return TriangularLogitTransport(params), diagnostics
    low, high = 0.0, 1.0
    for _ in range(int(bisection_iterations)):
        midpoint = (low + high) * 0.5
        midpoint_passed, _ = inspect(midpoint)
        if midpoint_passed:
            low = midpoint
        else:
            high = midpoint
    passed, diagnostics = inspect(low)
    if not passed:
        raise TriangularLogitTransportError(
            "identity-to-candidate safety search failed"
        )
    return TriangularLogitTransport(params, dose=low), diagnostics


__all__ = [
    "PARAMETER_COUNT",
    "TriangularLogitTransport",
    "TriangularLogitTransportError",
    "fit_triangular_logit_transport",
    "select_safe_transport",
    "transport_diagnostics",
]
