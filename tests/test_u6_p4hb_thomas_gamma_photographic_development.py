import json
from pathlib import Path

import numpy as np

from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_layer_support_matched_gamma_density,
    apply_layer_support_matched_thomas_gamma_density,
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
                ROOT / "outputs/u6_p2q_kodak_250d_characteristic_prior/bundle_run1.json"
            ).read_text()
        )["prior"]
    )
    return profile, prior


def test_p4hb_contract_changes_only_spatial_spectrum():
    contract = json.loads(
        (
            ROOT / "configs/u6_p4hb_thomas_gamma_photographic_development_v1.json"
        ).read_text()
    )
    candidate = contract["candidate"]
    assert candidate["marginals"].startswith("exact P4GZ")
    assert candidate["spatial_field"].startswith("exact P4BW")
    assert candidate["amplitude_multiplier"] == 1.0
    assert candidate["cohort_fitting_allowed"] is False
    assert contract["execution"]["same_cohort_rescue_allowed"] is False


def test_thomas_gamma_is_repeatable_bounded_and_spatially_distinct():
    profile, prior = _parents()
    base = np.full((37, 53, 3), (0.22, 0.48, 0.81), np.float32)
    seeds = (17, 29, 43)
    output, diagnostics = apply_layer_support_matched_thomas_gamma_density(
        base,
        profile=profile,
        prior=prior,
        layer_seeds=seeds,
        canonical_receipt_row_block_height=16,
    )
    repeated, repeated_diagnostics = apply_layer_support_matched_thomas_gamma_density(
        base,
        profile=profile,
        prior=prior,
        layer_seeds=seeds,
        canonical_receipt_row_block_height=16,
    )
    independent, _ = apply_layer_support_matched_gamma_density(
        base, profile=profile, prior=prior, layer_seeds=seeds
    )
    assert np.array_equal(output, repeated)
    assert diagnostics == repeated_diagnostics
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert diagnostics["limited_fraction"] == 0.0
    assert diagnostics["hard_clipping_used"] == 0.0
    assert len(set(diagnostics["receipt_ids"])) == 3
    assert not np.array_equal(output, independent)
