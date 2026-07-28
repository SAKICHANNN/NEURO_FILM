"""Clean-room baseline for the published two-matrix film-response equation.

This is an equation-family baseline, not a copy of the unavailable reference
implementation and not a calibrated film profile.  It intentionally returns
unclipped values so downstream evaluation can observe gamut failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares


EMULATING_EMULSION_BASELINE_SCHEMA = (
    "roll2film.clean_room_emulating_emulsion_equation.v1"
)


def _array(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class EmulatingEmulsionEquationOperator:
    """Two matrices around three independent four-parameter sigmoids."""

    capture_matrix: np.ndarray
    scan_matrix: np.ndarray
    response_amplitudes: np.ndarray
    response_slopes: np.ndarray
    response_midpoints: np.ndarray
    response_offsets: np.ndarray

    def __post_init__(self) -> None:
        capture = _array(
            self.capture_matrix, shape=(3, 3), name="capture_matrix"
        )
        scan = _array(self.scan_matrix, shape=(3, 3), name="scan_matrix")
        amplitudes = _array(
            self.response_amplitudes,
            shape=(3,),
            name="response_amplitudes",
        )
        slopes = _array(
            self.response_slopes, shape=(3,), name="response_slopes"
        )
        midpoints = _array(
            self.response_midpoints, shape=(3,), name="response_midpoints"
        )
        offsets = _array(
            self.response_offsets, shape=(3,), name="response_offsets"
        )
        if np.any(amplitudes <= 0.0) or np.any(slopes <= 0.0):
            raise ValueError("response amplitudes and slopes must be positive")
        object.__setattr__(self, "capture_matrix", capture)
        object.__setattr__(self, "scan_matrix", scan)
        object.__setattr__(self, "response_amplitudes", amplitudes)
        object.__setattr__(self, "response_slopes", slopes)
        object.__setattr__(self, "response_midpoints", midpoints)
        object.__setattr__(self, "response_offsets", offsets)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply the published equation literally, without output clipping."""

        values = np.asarray(rgb)
        if (
            values.ndim < 1
            or values.shape[-1] != 3
            or not np.issubdtype(values.dtype, np.number)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("rgb must be a finite numeric array ending in 3")
        flat = values.astype(np.float64, copy=False).reshape(-1, 3)
        layer_exposure = flat @ self.capture_matrix.T
        argument = self.response_slopes[None, :] * (
            layer_exposure - self.response_midpoints[None, :]
        )
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(argument, -80.0, 80.0)))
        response = (
            self.response_amplitudes[None, :] * sigmoid
            + self.response_offsets[None, :]
        )
        output = response @ self.scan_matrix.T
        return output.reshape(values.shape)

    def jacobian_determinants(self, rgb: np.ndarray) -> np.ndarray:
        """Return the analytic local Jacobian determinant at every RGB row."""

        values = np.asarray(rgb, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("rgb must be a finite Nx3 array")
        layer_exposure = values @ self.capture_matrix.T
        argument = self.response_slopes[None, :] * (
            layer_exposure - self.response_midpoints[None, :]
        )
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(argument, -80.0, 80.0)))
        derivatives = (
            self.response_amplitudes[None, :]
            * self.response_slopes[None, :]
            * sigmoid
            * (1.0 - sigmoid)
        )
        jacobians = np.einsum(
            "ij,njk,kl->nil",
            self.scan_matrix,
            np.eye(3)[None, :, :] * derivatives[:, :, None],
            self.capture_matrix,
        )
        return np.linalg.det(jacobians)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EMULATING_EMULSION_BASELINE_SCHEMA,
            "capture_matrix": self.capture_matrix.tolist(),
            "scan_matrix": self.scan_matrix.tolist(),
            "response_amplitudes": self.response_amplitudes.tolist(),
            "response_slopes": self.response_slopes.tolist(),
            "response_midpoints": self.response_midpoints.tolist(),
            "response_offsets": self.response_offsets.tolist(),
            "hard_output_clipping": False,
        }


@dataclass(frozen=True)
class EmulatingEmulsionFitResult:
    operator: EmulatingEmulsionEquationOperator
    development_rgb_rmse: float
    development_maximum_absolute_error: float
    optimization_cost: float
    optimality: float
    function_evaluations: int
    restart_index: int
    converged: bool


def _validate_pairs(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_rgb, dtype=np.float64)
    target = np.asarray(target_rgb, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or source.shape[0] < 12
        or target.shape != source.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("paired source and target must be finite Nx3 arrays, N>=12")
    return source, target


def _decode(parameters: np.ndarray) -> EmulatingEmulsionEquationOperator:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (30,) or not np.all(np.isfinite(values)):
        raise ValueError("candidate parameters must contain 30 finite values")
    return EmulatingEmulsionEquationOperator(
        capture_matrix=values[:9].reshape(3, 3),
        scan_matrix=values[9:18].reshape(3, 3),
        response_amplitudes=values[18:21],
        response_slopes=values[21:24],
        response_midpoints=values[24:27],
        response_offsets=values[27:30],
    )


def _initial_parameters(restart_index: int, seed: int) -> np.ndarray:
    values = np.concatenate(
        (
            np.eye(3).reshape(-1),
            np.eye(3).reshape(-1),
            np.ones(3),
            np.ones(3),
            np.full(3, 0.5),
            np.zeros(3),
        )
    ).astype(np.float64)
    if restart_index:
        rng = np.random.default_rng(seed + restart_index)
        values[:18] += rng.normal(0.0, 0.12, 18)
        values[18:21] += rng.normal(0.0, 0.10, 3)
        values[21:24] += rng.normal(0.0, 0.20, 3)
        values[24:27] += rng.normal(0.0, 0.20, 3)
        values[27:30] += rng.normal(0.0, 0.10, 3)
    return values


def fit_emulating_emulsion_equation(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    capture_matrix_entry_bounds: tuple[float, float] = (-2.0, 2.0),
    scan_matrix_entry_bounds: tuple[float, float] = (-2.0, 2.0),
    response_amplitude_bounds: tuple[float, float] = (0.01, 2.0),
    response_slope_bounds: tuple[float, float] = (0.05, 10.0),
    response_midpoint_bounds: tuple[float, float] = (-2.0, 2.0),
    response_offset_bounds: tuple[float, float] = (-1.0, 1.0),
    restart_count: int = 5,
    maximum_function_evaluations: int = 5000,
    function_tolerance: float = 1e-11,
    parameter_tolerance: float = 1e-11,
    gradient_tolerance: float = 1e-11,
    seed: int = 20250728,
) -> EmulatingEmulsionFitResult:
    """Fit the clean-room 30-parameter equation family by least squares."""

    source, target = _validate_pairs(source_rgb, target_rgb)
    bound_pairs = (
        capture_matrix_entry_bounds,
        scan_matrix_entry_bounds,
        response_amplitude_bounds,
        response_slope_bounds,
        response_midpoint_bounds,
        response_offset_bounds,
    )
    if (
        not isinstance(restart_count, int)
        or restart_count < 1
        or not isinstance(maximum_function_evaluations, int)
        or maximum_function_evaluations < 1
        or any(
            len(pair) != 2
            or not np.all(np.isfinite(pair))
            or float(pair[0]) >= float(pair[1])
            for pair in bound_pairs
        )
        or any(
            not np.isfinite(value) or value <= 0.0
            for value in (
                function_tolerance,
                parameter_tolerance,
                gradient_tolerance,
            )
        )
    ):
        raise ValueError("invalid clean-room fit controls")
    if response_amplitude_bounds[0] <= 0.0 or response_slope_bounds[0] <= 0.0:
        raise ValueError("amplitude and slope lower bounds must be positive")

    lower = np.concatenate(
        (
            np.full(9, capture_matrix_entry_bounds[0]),
            np.full(9, scan_matrix_entry_bounds[0]),
            np.full(3, response_amplitude_bounds[0]),
            np.full(3, response_slope_bounds[0]),
            np.full(3, response_midpoint_bounds[0]),
            np.full(3, response_offset_bounds[0]),
        )
    )
    upper = np.concatenate(
        (
            np.full(9, capture_matrix_entry_bounds[1]),
            np.full(9, scan_matrix_entry_bounds[1]),
            np.full(3, response_amplitude_bounds[1]),
            np.full(3, response_slope_bounds[1]),
            np.full(3, response_midpoint_bounds[1]),
            np.full(3, response_offset_bounds[1]),
        )
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (_decode(parameters).apply(source) - target).reshape(-1)

    best = None
    best_operator = None
    best_restart = -1
    for restart_index in range(restart_count):
        initial = np.clip(
            _initial_parameters(restart_index, seed),
            lower + 1e-9,
            upper - 1e-9,
        )
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            method="trf",
            max_nfev=maximum_function_evaluations,
            ftol=function_tolerance,
            xtol=parameter_tolerance,
            gtol=gradient_tolerance,
            x_scale="jac",
            loss="linear",
        )
        operator = _decode(result.x)
        if not np.all(np.isfinite(operator.apply(source))):
            continue
        if best is None or result.cost < best.cost:
            best = result
            best_operator = operator
            best_restart = restart_index
    if best is None or best_operator is None:
        raise ValueError("no finite clean-room equation fit was produced")

    error = best_operator.apply(source) - target
    return EmulatingEmulsionFitResult(
        operator=best_operator,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        development_maximum_absolute_error=float(np.max(np.abs(error))),
        optimization_cost=float(best.cost),
        optimality=float(best.optimality),
        function_evaluations=int(best.nfev),
        restart_index=best_restart,
        converged=bool(best.success),
    )


__all__ = [
    "EMULATING_EMULSION_BASELINE_SCHEMA",
    "EmulatingEmulsionEquationOperator",
    "EmulatingEmulsionFitResult",
    "fit_emulating_emulsion_equation",
]
