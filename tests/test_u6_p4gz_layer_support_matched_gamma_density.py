import json
from pathlib import Path

import numpy as np

from src.film_physics.density_conditioned_thomas import DensityConditionedThomasProfile
from src.film_physics.independent_density_nps import (
    apply_layer_support_matched_gamma_density,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior

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


def test_layer_gamma_is_repeatable_cube_safe_and_not_shared_ratio():
    profile, prior = _parents()
    base = np.full((63, 65, 3), (0.2, 0.4, 0.8), np.float64)
    first, diagnostics = apply_layer_support_matched_gamma_density(
        base, profile=profile, prior=prior, layer_seeds=(17, 23, 29)
    )
    second, repeated = apply_layer_support_matched_gamma_density(
        base, profile=profile, prior=prior, layer_seeds=(17, 23, 29)
    )
    assert np.array_equal(first, second)
    assert diagnostics == repeated
    assert np.all(first >= 0.0) and np.all(first <= 1.0)
    assert not np.allclose(first[..., 0] / first[..., 1], 0.5)
    assert diagnostics["limited_fraction"] == 0.0


def test_layer_gamma_exact_white_is_identity_per_layer():
    profile, prior = _parents()
    base = np.ones((31, 47, 3), np.float64)
    output, diagnostics = apply_layer_support_matched_gamma_density(
        base, profile=profile, prior=prior, layer_seeds=(17, 23, 29)
    )
    assert np.array_equal(output, base.astype(np.float32))
    assert diagnostics["support_degenerate_fraction"] == 1.0
    assert diagnostics["support_degenerate_residual_absolute"] == 0.0


def test_p4gz_contract_preserves_layer_separation_and_no_rescue():
    contract = json.loads(
        (
            ROOT / "configs/u6_p4gz_layer_support_matched_gamma_density_v1.json"
        ).read_text()
    )
    candidate = contract["candidate"]
    assert len(candidate["layer_field_seeds"]) == 3
    assert candidate["amplitude_multiplier"] == 1.0
    assert candidate["realized_variance_normalization_allowed"] is False
    assert candidate["clipping_allowed"] is False
