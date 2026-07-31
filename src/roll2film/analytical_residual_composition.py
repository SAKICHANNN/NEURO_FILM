"""Analytical cube-safe composition of one explicit RGB look over another."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factorized_boundary_guard import _maximum_safe_scale


@dataclass(frozen=True)
class AnalyticalResidualComposition:
    output: np.ndarray
    residual_scale: np.ndarray


def compose_analytical_residual(
    base_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    hard_boundary_epsilon: float,
    guard_boundary_epsilon: float,
) -> AnalyticalResidualComposition:
    base = np.asarray(base_rgb, dtype=np.float64)
    target = np.asarray(target_rgb, dtype=np.float64)
    if (
        base.shape != target.shape
        or base.ndim < 1
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(target))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or np.any(target < 0.0)
        or np.any(target > 1.0)
    ):
        raise ValueError("base and target must be matching finite [0,1] RGB")
    hard = float(hard_boundary_epsilon)
    guard = float(guard_boundary_epsilon)
    if (
        not np.isfinite(hard)
        or not np.isfinite(guard)
        or hard <= 0.0
        or guard <= hard
        or guard >= 0.5
    ):
        raise ValueError("invalid hard/guard boundary epsilon")

    lower = np.where(base <= hard, 0.0, np.minimum(base, guard))
    upper = np.where(base >= 1.0 - hard, 1.0, np.maximum(base, 1.0 - guard))
    delta = target - base
    scale = _maximum_safe_scale(base, delta, lower, upper)
    output = base + scale[..., None] * delta
    if (
        not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
    ):
        raise RuntimeError("analytical residual composition escaped RGB cube")
    return AnalyticalResidualComposition(output=output, residual_scale=scale)


__all__ = ["AnalyticalResidualComposition", "compose_analytical_residual"]
