"""Analytic sensitivity for the explicit density-to-print interpretation."""

from __future__ import annotations

import numpy as np

from src.roll2film.density_domain import density_to_print_reflectance
from src.roll2film.sensitometry_print import DensityToPrintInterpretation


def apply_density_to_print_with_jacobian(
    operator: DensityToPrintInterpretation,
    layer_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized print reflectance and d(output)/d(layer density).

    The Jacobian has shape ``(*layer_density.shape[:-1], 3, 3)`` with output
    channel before input channel.  It is the exact chain rule for the existing
    explicit operator; no finite difference or fitted surrogate is used.
    """
    if not isinstance(operator, DensityToPrintInterpretation):
        raise TypeError("operator must be DensityToPrintInterpretation")
    density = np.asarray(layer_density, dtype=np.float64)
    output = operator.apply(density)

    absorption = density @ operator.dye_absorption_matrix.T
    transmission = np.power(10.0, -absorption)
    transmission_jacobian = (
        -np.log(10.0)
        * transmission[..., :, None]
        * operator.dye_absorption_matrix
    )
    print_exposure = transmission @ operator.print_matrix.T
    exposure_jacobian = np.einsum(
        "ki,...ij->...kj", operator.print_matrix, transmission_jacobian
    )
    log_exposure_jacobian = exposure_jacobian / (
        (print_exposure + operator.exposure_floor)[..., :, None] * np.log(2.0)
    )
    log_exposure = np.log2(print_exposure + operator.exposure_floor)
    argument = operator.paper_slopes * (log_exposure - operator.paper_midpoints)
    sigmoid = 1.0 / (1.0 + np.exp(-argument))
    paper_density = operator.paper_maximum_densities * sigmoid
    paper_density_factor = (
        operator.paper_maximum_densities
        * operator.paper_slopes
        * sigmoid
        * (1.0 - sigmoid)
    )
    raw_reflectance = np.power(10.0, -paper_density)
    raw_jacobian = (
        -np.log(10.0)
        * raw_reflectance[..., :, None]
        * paper_density_factor[..., :, None]
        * log_exposure_jacobian
    )

    endpoints = density_to_print_reflectance(
        np.stack(
            (operator.black_reference_density, operator.white_reference_density),
            axis=0,
        ),
        operator.dye_absorption_matrix,
        operator.print_matrix,
        operator.paper_midpoints,
        operator.paper_slopes,
        operator.paper_maximum_densities,
        exposure_floor=operator.exposure_floor,
    )
    span = endpoints[1] - endpoints[0]
    jacobian = raw_jacobian / span[..., :, None]
    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError("print sensitivity is non-finite")
    return output, jacobian


__all__ = ["apply_density_to_print_with_jacobian"]
