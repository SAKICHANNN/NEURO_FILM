"""Replayable, numerically constrained explicit colour operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OPERATOR_SCHEMA = "roll2film.affine.v1"


@dataclass(frozen=True)
class AffineColorOperator:
    """An invertible global RGB affine operator.

    This deliberately small CT1 family is sufficient to falsify group-level
    identifiability before adding monotone curves or LUT residual capacity.
    Clipping is never implicit because it would destroy invertibility.
    """

    matrix: np.ndarray
    bias: np.ndarray
    working_space: str = "linear_srgb"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("matrix must have shape (3, 3)")
        if bias.shape != (3,):
            raise ValueError("bias must have shape (3,)")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
            raise ValueError("operator parameters must be finite")
        determinant = float(np.linalg.det(matrix))
        if determinant <= 1e-8:
            raise ValueError("matrix must be orientation-preserving and invertible")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bias", bias)

    @classmethod
    def identity(cls, working_space: str = "linear_srgb") -> "AffineColorOperator":
        return cls(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), working_space)

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.matrix))

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        return values @ self.matrix.T + self.bias

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        inverse_matrix = np.linalg.inv(self.matrix)
        return (values - self.bias) @ inverse_matrix.T

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPERATOR_SCHEMA,
            "working_space": self.working_space,
            "matrix": self.matrix.tolist(),
            "bias": self.bias.tolist(),
            "determinant": self.determinant,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AffineColorOperator":
        if payload.get("schema") != OPERATOR_SCHEMA:
            raise ValueError(f"unsupported operator schema: {payload.get('schema')!r}")
        return cls(
            matrix=np.asarray(payload["matrix"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
            working_space=str(payload["working_space"]),
        )


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("RGB values must have shape (..., 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("RGB values must be finite")
    return values
