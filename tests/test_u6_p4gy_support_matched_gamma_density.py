import json
from pathlib import Path

import numpy as np

from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_support_matched_gamma_density,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

ROOT = Path(__file__).resolve().parents[1]


def _parents():
    profile = DensityConditionedThomasProfile.from_dict(
        json.loads(
            (
                ROOT
                / "outputs/experiments/u6_p4bw_density_conditioned_thomas_profile_v1/run_a/bundle.json"
            ).read_text()
        )
    )
    prior = ManufacturerCharacteristicPrior.from_dict(
        json.loads(
            (
                ROOT
                / "outputs/u6_p2q_kodak_250d_characteristic_prior/bundle_run1.json"
            ).read_text()
        )["prior"]
    )
    return profile, prior


def test_support_matched_gamma_density_is_repeatable_and_cube_safe():
    profile, prior = _parents()
    base = np.full((63, 65, 3), (0.2, 0.4, 0.8), np.float64)
    first, diagnostics = apply_support_matched_gamma_density(
        base, profile=profile, prior=prior, seed=17
    )
    second, repeated = apply_support_matched_gamma_density(
        base, profile=profile, prior=prior, seed=17
    )
    assert np.array_equal(first, second)
    assert diagnostics == repeated
    assert np.all(first >= 0.0) and np.all(first <= 1.0)
    assert diagnostics["limited_fraction"] == 0.0
    assert diagnostics["hard_clipping_used"] == 0.0


def test_support_degenerate_white_is_exact_identity():
    profile, prior = _parents()
    base = np.ones((31, 47, 3), np.float64)
    output, diagnostics = apply_support_matched_gamma_density(
        base, profile=profile, prior=prior, seed=17
    )
    assert np.array_equal(output, base.astype(np.float32))
    assert diagnostics["support_degenerate_fraction"] == 1.0
    assert diagnostics["support_degenerate_residual_absolute"] == 0.0


def test_p4gy_contract_is_no_fit_no_clip_and_no_realized_normalization():
    contract = json.loads(
        (ROOT / "configs/u6_p4gy_support_matched_gamma_density_v1.json").read_text()
    )
    candidate = contract["candidate"]
    assert candidate["amplitude_multiplier"] == 1.0
    assert candidate["realized_variance_normalization_allowed"] is False
    assert candidate["clipping_allowed"] is False
