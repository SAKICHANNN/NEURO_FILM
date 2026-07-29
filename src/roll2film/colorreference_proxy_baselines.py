"""Low-capacity explicit baselines for the ColorReference paired proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import lsq_linear


ModelId = Literal[
    "identity",
    "target_mean",
    "diagonal_affine",
    "nonnegative_affine",
    "full_affine",
    "quadratic_full",
]


@dataclass(frozen=True)
class ProxyModel:
    model_id: ModelId
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.coefficients, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("proxy coefficients must be finite")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "coefficients", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "coefficients": self.coefficients.tolist(),
        }


def xyz_to_lab_d50(
    xyz: np.ndarray,
    *,
    reference_white: np.ndarray | tuple[float, float, float] = (
        0.9642,
        1.0,
        0.8251,
    ),
) -> np.ndarray:
    """Convert normalized XYZ to CIELAB without clipping."""

    values = np.asarray(xyz, dtype=np.float64)
    white = np.asarray(reference_white, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or white.shape != (3,)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(white))
        or np.any(white <= 0.0)
    ):
        raise ValueError("XYZ and reference white must be finite")
    ratio = values / white
    delta = 6.0 / 29.0
    threshold = delta**3
    transformed = np.where(
        ratio > threshold,
        np.cbrt(ratio),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * transformed[..., 1] - 16.0,
            500.0
            * (transformed[..., 0] - transformed[..., 1]),
            200.0
            * (transformed[..., 1] - transformed[..., 2]),
        ),
        axis=-1,
    )


def fit_proxy_model(
    source: np.ndarray,
    target_xyz: np.ndarray,
    *,
    model_id: ModelId,
    quadratic_ridge: float,
    nonnegative_maximum_iterations: int,
) -> ProxyModel:
    """Fit one fixed explicit baseline on development rows only."""

    x, y = _validate_pairs(source, target_xyz)
    if model_id == "identity":
        return ProxyModel(model_id, np.empty((0,), dtype=np.float64))
    if model_id == "target_mean":
        return ProxyModel(model_id, y.mean(axis=0))
    if model_id == "diagonal_affine":
        coefficients = np.empty((3, 2), dtype=np.float64)
        for channel in range(3):
            design = np.column_stack(
                (x[:, channel], np.ones(x.shape[0]))
            )
            coefficients[channel] = np.linalg.lstsq(
                design, y[:, channel], rcond=None
            )[0]
        return ProxyModel(model_id, coefficients)
    if model_id in ("full_affine", "nonnegative_affine"):
        design = np.column_stack((x, np.ones(x.shape[0])))
        if model_id == "full_affine":
            coefficients = np.linalg.lstsq(
                design, y, rcond=None
            )[0]
        else:
            columns = []
            for channel in range(3):
                result = lsq_linear(
                    design,
                    y[:, channel],
                    bounds=(0.0, np.inf),
                    max_iter=nonnegative_maximum_iterations,
                    lsmr_tol="auto",
                    verbose=0,
                )
                if not result.success:
                    raise RuntimeError(
                        "nonnegative affine fit did not converge"
                    )
                columns.append(result.x)
            coefficients = np.stack(columns, axis=1)
        return ProxyModel(model_id, coefficients)
    if model_id == "quadratic_full":
        if (
            not np.isfinite(quadratic_ridge)
            or quadratic_ridge <= 0.0
        ):
            raise ValueError("quadratic ridge must be positive")
        design = quadratic_basis(x)
        regularizer = np.eye(design.shape[1], dtype=np.float64)
        regularizer[0, 0] = 0.0
        normal = (
            design.T @ design
            + quadratic_ridge * regularizer
        )
        coefficients = np.linalg.solve(normal, design.T @ y)
        return ProxyModel(model_id, coefficients)
    raise ValueError(f"unsupported proxy model {model_id}")


def apply_proxy_model(
    model: ProxyModel, source: np.ndarray
) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("source must be finite Nx3")
    if model.model_id == "identity":
        return values.copy()
    if model.model_id == "target_mean":
        return np.broadcast_to(
            model.coefficients, values.shape
        ).copy()
    if model.model_id == "diagonal_affine":
        return (
            values * model.coefficients[:, 0][None, :]
            + model.coefficients[:, 1][None, :]
        )
    if model.model_id in ("full_affine", "nonnegative_affine"):
        design = np.column_stack(
            (values, np.ones(values.shape[0]))
        )
        return design @ model.coefficients
    if model.model_id == "quadratic_full":
        return quadratic_basis(values) @ model.coefficients
    raise ValueError(f"unsupported proxy model {model.model_id}")


def quadratic_basis(source: np.ndarray) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("quadratic source must be Nx3")
    red, green, blue = values.T
    return np.column_stack(
        (
            np.ones(values.shape[0]),
            red,
            green,
            blue,
            red * red,
            green * green,
            blue * blue,
            red * green,
            red * blue,
            green * blue,
        )
    )


def _validate_pairs(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != 3
        or y.shape != x.shape
        or x.shape[0] < 12
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("proxy pairs must be finite matching Nx3 arrays")
    return x, y
