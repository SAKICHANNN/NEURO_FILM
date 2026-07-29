from __future__ import annotations

import numpy as np

from src.eval.real_uniform_grain_physical import (
    render_anisotropic_structure,
)


def _render(
    target: np.ndarray,
    *,
    origin_yx: tuple[int, int] | None = None,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return render_anisotropic_structure(
        target,
        grain_optical_density_by_channel=[0.04, 0.05, 0.035],
        sigma_yx=(0.9, 0.65),
        seeds=[11, 13, 17],
        maximum_target_density=2.0,
        truncate=4.0,
        origin_yx=origin_yx,
        shape=shape,
    )


def test_anisotropic_structure_preserves_zero_and_physical_domains() -> None:
    target = np.zeros((64, 72, 3), dtype=np.float64)
    density, transmittance = _render(target)
    assert np.array_equal(density, np.zeros_like(density))
    assert np.array_equal(transmittance, np.ones_like(transmittance))
    target.fill(0.8)
    density, transmittance = _render(target)
    assert np.min(density) >= 0.0
    assert np.min(transmittance) > 0.0
    assert np.max(transmittance) <= 1.0


def test_anisotropic_structure_is_repeatable_and_partition_exact() -> None:
    ramp = np.linspace(0.0, 1.8, 83, dtype=np.float64)
    target = np.broadcast_to(ramp[None, :, None], (79, 83, 3)).copy()
    full_density, full_transmittance = _render(target)
    repeat_density, repeat_transmittance = _render(target)
    assert np.array_equal(full_density, repeat_density)
    assert np.array_equal(full_transmittance, repeat_transmittance)
    density_rows = []
    transmittance_rows = []
    for y0 in range(0, target.shape[0], 17):
        height = min(17, target.shape[0] - y0)
        density, transmittance = _render(
            target,
            origin_yx=(y0, 0),
            shape=(height, target.shape[1]),
        )
        density_rows.append(density)
        transmittance_rows.append(transmittance)
    assert np.array_equal(full_density, np.concatenate(density_rows, axis=0))
    assert np.array_equal(
        full_transmittance,
        np.concatenate(transmittance_rows, axis=0),
    )
