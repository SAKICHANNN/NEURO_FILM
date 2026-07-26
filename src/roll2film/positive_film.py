"""Compact clean-room positive-film response colour operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .density_domain import _readonly_matrix, _readonly_vector, logistic_density


POSITIVE_FILM_RESPONSE_SCHEMA = "roll2film.positive_film_response.v1"


@dataclass(frozen=True)
class PositiveFilmResponseOperator:
    """Two colour matrices around per-layer log-exposure sigmoid responses."""

    capture_matrix: np.ndarray
    response_midpoints: np.ndarray
    response_slopes: np.ndarray
    maximum_responses: np.ndarray
    scan_matrix: np.ndarray
    exposure_floor: float = 2.0**-16
    working_space: str = "linear_srgb_d65"
    matrix_minimum_determinant: float = 0.2
    minimum_endpoint_span: float = 0.05

    def __post_init__(self) -> None:
        for field in ("capture_matrix", "scan_matrix"):
            matrix = _readonly_matrix(getattr(self, field), field)
            if np.any(matrix < 0.0):
                raise ValueError(f"{field} must be non-negative")
            if np.max(np.abs(matrix.sum(axis=1) - 1.0)) > 1e-12:
                raise ValueError(f"{field} must be row-stochastic")
            if float(np.linalg.det(matrix)) < self.matrix_minimum_determinant:
                raise ValueError(f"{field} determinant is below the contract")
            object.__setattr__(self, field, matrix)
        for field in (
            "response_midpoints",
            "response_slopes",
            "maximum_responses",
        ):
            object.__setattr__(self, field, _readonly_vector(getattr(self, field), field))
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 positive-film operator requires linear_srgb_d65")
        if not np.isfinite(self.exposure_floor) or self.exposure_floor != 2.0**-16:
            raise ValueError("v1 exposure_floor must equal 2^-16")
        if not 0.0 < self.matrix_minimum_determinant < 1.0:
            raise ValueError("matrix_minimum_determinant must be in (0, 1)")
        if not np.isfinite(self.minimum_endpoint_span) or self.minimum_endpoint_span <= 0.0:
            raise ValueError("minimum_endpoint_span must be finite and positive")
        if np.any(self.response_midpoints < -12.0) or np.any(
            self.response_midpoints > 2.0
        ):
            raise ValueError("log2 midpoints must be in [-12, 2]")
        if np.any(self.response_slopes < 0.2) or np.any(self.response_slopes > 4.0):
            raise ValueError("response slopes must be in [0.2, 4.0]")
        if np.any(self.maximum_responses < 0.2) or np.any(
            self.maximum_responses > 4.0
        ):
            raise ValueError("maximum responses must be in [0.2, 4.0]")
        black, white = self._raw_endpoints()
        span = white - black
        if np.any(span < self.minimum_endpoint_span):
            raise ValueError(
                f"theoretical endpoint span is below the contract: {span.tolist()}"
            )

    def _raw_response(self, linear_rgb: np.ndarray) -> np.ndarray:
        layer_exposure = linear_rgb @ self.capture_matrix.T
        log_exposure = np.log2(layer_exposure + self.exposure_floor)
        layer_response = logistic_density(
            log_exposure,
            self.response_midpoints,
            self.response_slopes,
            self.maximum_responses,
        )
        return layer_response @ self.scan_matrix.T

    def _raw_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        values = self._raw_response(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        )
        return values[0], values[1]

    @property
    def endpoint_span(self) -> np.ndarray:
        black, white = self._raw_endpoints()
        return white - black

    def apply(self, linear_rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        rgb = np.asarray(linear_rgb, dtype=np.float64)
        if (
            rgb.ndim < 2
            or rgb.shape[-1] != 3
            or not np.all(np.isfinite(rgb))
            or np.any(rgb < 0.0)
            or np.any(rgb > 1.0)
        ):
            raise ValueError("linear_rgb must be finite [0, 1] data with shape (..., 3)")
        if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        black, white = self._raw_endpoints()
        full = (self._raw_response(rgb) - black) / (white - black)
        if np.any(full < -1e-12) or np.any(full > 1.0 + 1e-12):
            raise RuntimeError("positive-film operator escaped theoretical endpoints")
        if strength == 0.0:
            return rgb.copy()
        if strength == 1.0:
            return full
        return rgb + strength * (full - rgb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POSITIVE_FILM_RESPONSE_SCHEMA,
            "working_space": self.working_space,
            "exposure_floor": self.exposure_floor,
            "matrix_minimum_determinant": self.matrix_minimum_determinant,
            "minimum_endpoint_span": self.minimum_endpoint_span,
            "capture_matrix": self.capture_matrix.tolist(),
            "response_midpoints": self.response_midpoints.tolist(),
            "response_slopes": self.response_slopes.tolist(),
            "maximum_responses": self.maximum_responses.tolist(),
            "scan_matrix": self.scan_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PositiveFilmResponseOperator":
        if payload.get("schema") != POSITIVE_FILM_RESPONSE_SCHEMA:
            raise ValueError("unsupported positive-film operator schema")
        return cls(
            capture_matrix=np.asarray(payload["capture_matrix"], dtype=np.float64),
            response_midpoints=np.asarray(payload["response_midpoints"], dtype=np.float64),
            response_slopes=np.asarray(payload["response_slopes"], dtype=np.float64),
            maximum_responses=np.asarray(payload["maximum_responses"], dtype=np.float64),
            scan_matrix=np.asarray(payload["scan_matrix"], dtype=np.float64),
            exposure_floor=float(payload["exposure_floor"]),
            working_space=str(payload["working_space"]),
            matrix_minimum_determinant=float(payload["matrix_minimum_determinant"]),
            minimum_endpoint_span=float(payload["minimum_endpoint_span"]),
        )


def positive_film_operator_from_config(
    payload: dict[str, Any],
    *,
    exposure_floor: float = 2.0**-16,
    matrix_minimum_determinant: float = 0.2,
    minimum_endpoint_span: float = 0.05,
) -> PositiveFilmResponseOperator:
    return PositiveFilmResponseOperator(
        capture_matrix=np.asarray(payload["capture_matrix"], dtype=np.float64),
        response_midpoints=np.asarray(payload["response_midpoints"], dtype=np.float64),
        response_slopes=np.asarray(payload["response_slopes"], dtype=np.float64),
        maximum_responses=np.asarray(payload["maximum_responses"], dtype=np.float64),
        scan_matrix=np.asarray(payload["scan_matrix"], dtype=np.float64),
        exposure_floor=exposure_floor,
        matrix_minimum_determinant=matrix_minimum_determinant,
        minimum_endpoint_span=minimum_endpoint_span,
    )


def finite_difference_jacobians(
    operator: PositiveFilmResponseOperator,
    points: np.ndarray,
    *,
    step: float,
    strength: float = 1.0,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or not np.isfinite(step)
        or step <= 0.0
        or np.any(values < step)
        or np.any(values > 1.0 - step)
    ):
        raise ValueError("points must be finite interior RGB rows for the given step")
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        forward = operator.apply(values + offset, strength=strength)
        backward = operator.apply(values - offset, strength=strength)
        columns.append((forward - backward) / (2.0 * step))
    return np.stack(columns, axis=-1)


__all__ = [
    "POSITIVE_FILM_RESPONSE_SCHEMA",
    "PositiveFilmResponseOperator",
    "finite_difference_jacobians",
    "positive_film_operator_from_config",
]
