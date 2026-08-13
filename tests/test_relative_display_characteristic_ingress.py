from pathlib import Path

import numpy as np
import pytest

from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.relative_display_characteristic_ingress import (
    relative_display_to_finite_density_transmittance,
)

ROOT = Path(__file__).resolve().parents[1]


def _prior():
    import json

    path = (
        ROOT
        / "outputs/experiments/u6_p4hk_bounded_photographic_profile_bundle_v1/run_a/profile.json"
    )
    return reconstruct_bounded_photographic_profile(
        json.loads(path.read_text(encoding="utf-8"))
    ).prior


def test_ingress_maps_zero_and_one_to_finite_curve_endpoints() -> None:
    values = np.stack([np.zeros((3, 3)), np.ones((3, 3))])
    density, transmittance, receipt = relative_display_to_finite_density_transmittance(
        values, _prior()
    )
    assert np.all(np.isfinite(density))
    assert np.all(transmittance > 0.0)
    assert np.all(transmittance <= 1.0)
    assert np.all(density[1] >= density[0])
    assert np.all(transmittance[1] <= transmittance[0])
    assert receipt["calibrated_exposure_claimed"] is False


def test_ingress_is_monotone_per_layer() -> None:
    ramp = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    values = np.repeat(ramp[:, None], 3, axis=1)
    density, transmittance, _ = relative_display_to_finite_density_transmittance(
        values, _prior()
    )
    assert np.all(np.diff(density, axis=0) >= 0.0)
    assert np.all(np.diff(transmittance, axis=0) <= 0.0)


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan])
def test_ingress_rejects_out_of_domain(bad: float) -> None:
    with pytest.raises(ValueError):
        relative_display_to_finite_density_transmittance(
            np.full((2, 2, 3), bad), _prior()
        )
