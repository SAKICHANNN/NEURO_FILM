from __future__ import annotations

import numpy as np
import pytest

from src.eval.physical_silver_photographic import apply_bounded_silver_density
from src.film_physics.silver_retention import SilverRetentionProfile


def _profile() -> SilverRetentionProfile:
    return SilverRetentionProfile(
        0.35, (0.2126, 0.7152, 0.0722), 3.0, 4.05
    )


def test_density_headroom_is_analytical_and_unclipped() -> None:
    rng = np.random.default_rng(41)
    density = rng.uniform(0.1, 1.4, size=(31, 43, 3))
    white = np.array([1.56, 1.49, 1.43])
    output, scale = apply_bounded_silver_density(
        density, _profile(), white
    )
    assert np.all(output >= density)
    assert np.all(output <= white)
    assert np.all((scale >= 0.0) & (scale <= 1.0))
    assert np.any(scale < 1.0)


def test_out_of_domain_density_fails_closed() -> None:
    density = np.full((3, 5, 3), 1.0)
    density[0, 0, 1] = 1.6
    with pytest.raises(ValueError, match="headroom"):
        apply_bounded_silver_density(
            density, _profile(), np.array([1.56, 1.49, 1.43])
        )
