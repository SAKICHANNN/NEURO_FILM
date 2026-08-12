import json
from pathlib import Path

import numpy as np

from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_cross_layer_thomas_gamma_copula,
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


def test_p4hc_copula_is_repeatable_bounded_and_correlated():
    profile, prior = _parents()
    contract = json.loads(
        (ROOT / "configs/u6_p4hc_cross_layer_thomas_gamma_copula_v1.json").read_text()
    )
    correlation = np.asarray(
        contract["candidate"]["correlation_matrix"], dtype=np.float64
    )
    base = np.full((41, 59, 3), (0.22, 0.48, 0.81), np.float32)
    arguments = {
        "profile": profile,
        "prior": prior,
        "layer_seeds": (17, 29, 43),
        "correlation_matrix": correlation,
        "canonical_receipt_row_block_height": 16,
    }
    output, diagnostics = apply_cross_layer_thomas_gamma_copula(base, **arguments)
    repeated, repeated_diagnostics = apply_cross_layer_thomas_gamma_copula(
        base, **arguments
    )
    empirical = np.asarray(
        diagnostics["empirical_copula_correlation"], dtype=np.float64
    )
    assert np.array_equal(output, repeated)
    assert diagnostics == repeated_diagnostics
    assert np.max(np.abs(empirical - correlation)) < 0.04
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert diagnostics["limited_fraction"] == 0.0
    assert diagnostics["hard_clipping_used"] == 0.0


def test_p4hc_rejects_non_positive_definite_correlation():
    profile, prior = _parents()
    base = np.full((9, 11, 3), 0.5, np.float32)
    bad = np.asarray([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    try:
        apply_cross_layer_thomas_gamma_copula(
            base,
            profile=profile,
            prior=prior,
            layer_seeds=(1, 2, 3),
            correlation_matrix=bad,
            canonical_receipt_row_block_height=4,
        )
    except ValueError as error:
        assert "positive definite" in str(error)
    else:
        raise AssertionError("non-positive-definite correlation was accepted")
