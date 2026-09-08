from __future__ import annotations

from itertools import product

import numpy as np


def cell_vertex_jacobians(cube: np.ndarray) -> np.ndarray:
    values = np.asarray(cube)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or len(set(values.shape[:3])) != 1
        or values.shape[0] < 2
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected finite floating [B,G,R,output_RGB] cubic nodes")
    values = values.astype(np.float64)
    count = values.shape[0] - 1
    dr = np.diff(values, axis=2) * count
    dg = np.diff(values, axis=1) * count
    db = np.diff(values, axis=0) * count
    corners = []
    for b, g, r in product((0, 1), repeat=3):
        # Each derivative is from inside this cell, including shared boundaries.
        corners.append(
            np.stack(
                (
                    dr[b : b + count, g : g + count, :],
                    dg[b : b + count, :, r : r + count],
                    db[:, g : g + count, r : r + count],
                ),
                axis=-1,
            )
        )
    result = np.stack(corners, axis=3)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite float64 derivative")
    return result


def numerical_cell_bounds(
    cube: np.ndarray, *, absolute_margin: float, relative_margin: float
) -> tuple[np.ndarray, dict]:
    margins = np.asarray([absolute_margin, relative_margin], dtype=np.float64)
    if not np.isfinite(margins).all() or (margins < 0).any():
        raise ValueError("Expected finite nonnegative numerical margins")
    jac = cell_vertex_jacobians(cube)
    gain = np.linalg.svd(jac, compute_uv=False)[..., 0]
    residual = np.linalg.svd(jac - np.eye(3), compute_uv=False)[..., 0]
    cell_gain = gain.max(axis=3)
    cell_residual = residual.max(axis=3)
    gain_upper = cell_gain + absolute_margin + relative_margin * cell_gain
    residual_upper = cell_residual + absolute_margin + relative_margin * cell_residual
    lower = np.maximum(0.0, 1.0 - residual_upper)
    bounds = np.stack(
        (cell_gain, cell_residual, gain_upper, residual_upper, lower), axis=-1
    )
    determinants = np.linalg.det(jac)
    if not np.isfinite(bounds).all() or not np.isfinite(determinants).all():
        raise ValueError("Nonfinite float64 numerical bound or determinant")
    residual_max = float(residual_upper.max())
    summary = {
        "cell_shape_bgr": list(cell_gain.shape),
        "cell_count": int(cell_gain.size),
        "one_sided_vertex_count": int(gain.size),
        "vertex_order_bgr": [list(v) for v in product((0, 1), repeat=3)],
        "matrix_axes": ["output_RGB", "input_RGB"],
        "cell_array_columns": [
            "maximum_vertex_spectral_norm",
            "maximum_vertex_residual_spectral_norm",
            "gain_upper_numerical_estimate_with_margin",
            "residual_upper_numerical_estimate_with_margin",
            "singular_lower_sufficient_numerical_estimate",
        ],
        "maximum_vertex_spectral_norm": float(gain.max()),
        "maximum_vertex_residual_spectral_norm": float(residual.max()),
        "global_gain_upper_numerical_estimate": float(gain_upper.max()),
        "global_residual_upper_numerical_estimate": residual_max,
        "global_singular_lower_sufficient_numerical_estimate": float(lower.min()),
        "global_residual_condition": (
            "NUMERICALLY_SATISFIED_SUFFICIENT_CONDITION"
            if residual_max < 1.0
            else "INCONCLUSIVE"
        ),
        "cells_satisfying_residual_sufficient_condition": int(
            (residual_upper < 1.0).sum()
        ),
        "maximum_gain_cell_bgr": list(
            map(int, np.unravel_index(np.argmax(cell_gain), cell_gain.shape))
        ),
        "maximum_residual_cell_bgr": list(
            map(int, np.unravel_index(np.argmax(cell_residual), cell_residual.shape))
        ),
        "vertex_determinant_samples_only": {
            "minimum": float(determinants.min()),
            "maximum": float(determinants.max()),
            "negative_count": int((determinants < 0).sum()),
            "zero_count": int((determinants == 0).sum()),
            "count": int(determinants.size),
            "full_cell_sign_certified": False,
        },
        "numerical_margin": {
            "absolute": absolute_margin,
            "relative": relative_margin,
        },
        "rounding_error_certified": False,
        "interpretation": (
            "Exact-real trilinear cell Jacobians are convex combinations of their "
            "one-sided vertex Jacobians. Spectral-norm convexity bounds both J "
            "and J-I throughout each cell. A global residual bound below one "
            "implies an injective bi-Lipschitz cube map in exact arithmetic; "
            "the lower bound here is sufficient, not necessary. Float64 SVD "
            "and added margins are numerical estimates, not interval proofs."
        ),
        "continuous_interpolation_only": True,
        "photographic_safety_or_quality_claim": False,
    }
    return bounds, summary
