"""Independent zero-DC density NPS texture for display-scale development."""

from __future__ import annotations

import numpy as np

from src.film_physics.structure_compiler import counter_normal_region

_KERNEL = np.asarray([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float64)


def _aperture(values: np.ndarray) -> np.ndarray:
    padded_x = np.pad(values, ((0, 0), (2, 2)), mode="edge")
    horizontal = sum(
        weight * padded_x[:, offset : offset + values.shape[1]]
        for offset, weight in enumerate(_KERNEL)
    )
    padded_y = np.pad(horizontal, ((2, 2), (0, 0)), mode="edge")
    return sum(
        weight * padded_y[offset : offset + values.shape[0], :]
        for offset, weight in enumerate(_KERNEL)
    )


def synthesize_independent_density_nps(
    shape: tuple[int, int], *, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Return one exact-zero-DC, exact-unit-RMS counter-seeded NPS field."""

    if (
        len(shape) != 2
        or any(not isinstance(value, int) or value < 5 for value in shape)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("invalid independent density NPS request")
    white = counter_normal_region(shape, origin_yx=(0, 0), shape=shape, seed=seed)
    field = white - _aperture(white)
    field -= float(np.mean(field, dtype=np.float64))
    standard_deviation = float(np.std(field, dtype=np.float64))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise RuntimeError("independent density NPS field is degenerate")
    field /= standard_deviation
    field = np.ascontiguousarray(field, dtype=np.float64)
    return field, {
        "field_mean": float(np.mean(field, dtype=np.float64)),
        "field_std": float(np.std(field, dtype=np.float64)),
    }


def apply_independent_density_nps(
    neutral_base: np.ndarray, *, density_sigma: float, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a bounded common density NPS field with an analytic cube guard."""

    base = np.asarray(neutral_base, dtype=np.float64)
    if (
        base.ndim != 3
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or not np.isfinite(density_sigma)
        or density_sigma <= 0.0
    ):
        raise ValueError("invalid independent density NPS input")
    field, diagnostics = synthesize_independent_density_nps(base.shape[:2], seed=seed)
    finite_tail = 3.0 * np.tanh(field / 3.0)
    luminance = 0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2]
    visibility = 4.0 * luminance * (1.0 - luminance)
    requested_density = density_sigma * visibility * finite_tail

    scale = np.ones_like(requested_density)
    brightening = requested_density < 0.0
    if np.any(brightening):
        positive_base = base > 0.0
        allowed = np.full_like(base, np.inf)
        allowed[positive_base] = np.log10(
            (1.0 - 16.0 * np.finfo(np.float64).eps) / base[positive_base]
        )
        allowed_brightening = np.min(allowed, axis=-1)
        scale[brightening] = np.minimum(
            1.0,
            allowed_brightening[brightening]
            / np.maximum(-requested_density[brightening], np.finfo(np.float64).tiny),
        )
    scale = np.clip(scale, 0.0, 1.0)
    output64 = base * np.power(10.0, -(requested_density * scale))[..., None]
    output = np.ascontiguousarray(output64, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("independent density NPS escaped its output domain")
    residual = output.astype(np.float64) - base
    diagnostics.update(
        {
            "finite_tail_mean": float(np.mean(finite_tail, dtype=np.float64)),
            "finite_tail_std": float(np.std(finite_tail, dtype=np.float64)),
            "mean_density_visibility": float(np.mean(visibility, dtype=np.float64)),
            "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "minimum_residual_scale": float(np.min(scale)),
            "limited_fraction": float(np.mean(scale < 1.0)),
            "hard_clipping_used": 0.0,
        }
    )
    return output, diagnostics


__all__ = ["apply_independent_density_nps", "synthesize_independent_density_nps"]
