from __future__ import annotations

import numpy as np

from src.eval.real_uniform_grain_shaping import anisotropic_poisson_region


def test_anisotropic_poisson_region_is_repeatable_and_partition_exact() -> None:
    kwargs = {
        "full_shape": (131, 137),
        "rate": 16.0,
        "sigma_yx": (0.45, 0.75),
        "seed": 260901,
        "truncate": 4.0,
    }
    full = anisotropic_poisson_region(
        **kwargs,
        origin_yx=(0, 0),
        shape=kwargs["full_shape"],
    )
    repeat = anisotropic_poisson_region(
        **kwargs,
        origin_yx=(0, 0),
        shape=kwargs["full_shape"],
    )
    stitched = np.concatenate(
        [
            anisotropic_poisson_region(
                **kwargs,
                origin_yx=(y0, 0),
                shape=(min(31, kwargs["full_shape"][0] - y0), 137),
            )
            for y0 in range(0, kwargs["full_shape"][0], 31)
        ],
        axis=0,
    )
    assert np.array_equal(full, repeat)
    assert np.array_equal(full, stitched)
    assert np.all(np.isfinite(full))
    assert np.min(full) >= 0.0


def test_anisotropic_poisson_region_rejects_invalid_geometry() -> None:
    try:
        anisotropic_poisson_region(
            full_shape=(64, 64),
            origin_yx=(60, 0),
            shape=(8, 8),
            rate=1.0,
            sigma_yx=(0.5, 0.5),
            seed=1,
            truncate=4.0,
        )
    except ValueError as error:
        assert "invalid anisotropic" in str(error)
    else:
        raise AssertionError("out-of-bounds request must fail")
