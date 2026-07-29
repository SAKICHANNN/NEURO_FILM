from __future__ import annotations

import numpy as np

from src.real_film.it8_density_basis import (
    fit_density_basis,
    optical_density,
    reconstruct_density,
    reconstruction_rmse,
)


def test_optical_density_is_bounded_and_counts_measurement_floor() -> None:
    values = np.asarray([[100.0, 10.0, 0.0, -0.1]])
    density, floor_count = optical_density(values, floor_percent=0.01)
    assert floor_count == 2
    assert density.tolist() == [[0.0, 1.0, 4.0, 4.0]]


def test_rank_two_basis_reconstructs_nonnegative_low_rank_fixture() -> None:
    weights = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    components = np.asarray([[1.0, 0.5, 0.0], [0.0, 0.2, 1.0]])
    values = weights @ components
    fitted, model = fit_density_basis(
        values,
        rank=2,
        init="nndsvda",
        solver="cd",
        tolerance=1e-7,
        maximum_iterations=5000,
        random_state=2909,
    )
    assert fitted.converged
    assert fitted.components.shape == (2, 3)
    assert reconstruct_density(model, values).shape == values.shape
    assert reconstruction_rmse(model, values) < 1e-4
