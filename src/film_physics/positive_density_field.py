"""Intrinsically positive developed-density field transforms."""

from __future__ import annotations

import math

import numpy as np


def softplus_density_field(unit_field: np.ndarray, *, a: float, b: float) -> np.ndarray:
    unit = np.asarray(unit_field, dtype=np.float64)
    if (
        unit.ndim != 2
        or not unit.size
        or not np.all(np.isfinite(unit))
        or not math.isfinite(a)
        or not math.isfinite(b)
        or b < 0.0
    ):
        raise ValueError("invalid positive density-field inputs")
    result = np.ascontiguousarray(np.logaddexp(0.0, a + b * unit), dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError("softplus density field is not finite and positive")
    result.setflags(write=False)
    return result


__all__ = ["softplus_density_field"]
