"""Offline nonnegative spectral-density bases for ColorReference targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from sklearn.decomposition import NMF


@dataclass(frozen=True)
class FittedDensityBasis:
    rank: int
    components: np.ndarray
    reconstruction_error: float
    iterations: int
    maximum_iterations: int

    @property
    def converged(self) -> bool:
        return self.iterations < self.maximum_iterations

    @property
    def components_sha256(self) -> str:
        canonical = np.ascontiguousarray(self.components, dtype="<f8")
        return hashlib.sha256(canonical.tobytes()).hexdigest()


def optical_density(
    transmittance_percent: np.ndarray,
    *,
    floor_percent: float,
) -> tuple[np.ndarray, int]:
    values = np.asarray(transmittance_percent, dtype=np.float64)
    if values.ndim < 2 or not np.all(np.isfinite(values)):
        raise ValueError("spectral transmittance must be finite and at least 2D")
    if floor_percent <= 0.0 or floor_percent >= 100.0:
        raise ValueError("invalid transmittance floor")
    floored = values < floor_percent
    density = -np.log10(np.maximum(values, floor_percent) / 100.0)
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise RuntimeError("optical density left the nonnegative finite domain")
    return density, int(np.count_nonzero(floored))


def fit_density_basis(
    rows: np.ndarray,
    *,
    rank: int,
    init: str,
    solver: str,
    tolerance: float,
    maximum_iterations: int,
    random_state: int,
) -> tuple[FittedDensityBasis, NMF]:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("basis rows must be a finite matrix")
    if np.any(values < 0.0):
        raise ValueError("basis rows must be nonnegative optical density")
    model = NMF(
        n_components=rank,
        init=init,
        solver=solver,
        beta_loss="frobenius",
        tol=tolerance,
        max_iter=maximum_iterations,
        random_state=random_state,
        shuffle=False,
    ).fit(values)
    fitted = FittedDensityBasis(
        rank=rank,
        components=np.asarray(model.components_, dtype=np.float64),
        reconstruction_error=float(model.reconstruction_err_),
        iterations=int(model.n_iter_),
        maximum_iterations=maximum_iterations,
    )
    return fitted, model


def reconstruction_rmse(model: NMF, rows: np.ndarray) -> float:
    reconstructed = reconstruct_density(model, rows)
    values = np.asarray(rows, dtype=np.float64)
    return float(np.sqrt(np.mean((values - reconstructed) ** 2)))


def reconstruct_density(model: NMF, rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    weights = model.transform(values)
    return weights @ model.components_
