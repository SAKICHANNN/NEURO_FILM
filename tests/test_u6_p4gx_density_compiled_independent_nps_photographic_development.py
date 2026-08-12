import hashlib
import json
from pathlib import Path

import numpy as np

from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_density_compiled_independent_nps,
    apply_independent_density_nps,
    compile_conservative_shared_sigma_d,
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


def test_p4gx_contract_freezes_observed_amplitude_without_scaling():
    contract = json.loads(
        (
            ROOT
            / "configs/u6_p4gx_density_compiled_independent_nps_photographic_development_v1.json"
        ).read_text()
    )
    candidate = contract["candidate"]
    assert candidate["amplitude_multiplier"] == 1.0
    assert candidate["cohort_fitting_allowed"] is False
    assert contract["execution"]["same_cohort_rescue_allowed"] is False
    assert contract["automatic_gates"] == json.loads(
        (
            ROOT
            / "configs/u6_p4gw_independent_density_nps_photographic_development_v1.json"
        ).read_text()
    )["automatic_gates"]


def test_compiled_shared_sigma_is_layer_minimum_and_in_observed_range():
    profile, prior = _parents()
    base = np.asarray(
        [[[0.0, 0.0, 0.0], [0.2, 0.4, 0.8], [1.0, 1.0, 1.0]]],
        dtype=np.float64,
    )
    shared, diagnostics = compile_conservative_shared_sigma_d(
        base, profile=profile, prior=prior
    )
    luminance = 0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2]
    layers = []
    for index, channel in enumerate(("red", "green", "blue")):
        lower, upper = prior.curves[index].domain
        exposure = lower + luminance * (upper - lower)
        layers.append(
            profile.amplitude_profile.evaluate_channel(prior, channel, exposure)
        )
    assert np.array_equal(shared, np.min(np.stack(layers, axis=-1), axis=-1))
    assert 0.004 <= diagnostics["minimum_compiled_sigma_d"] < 0.012
    assert diagnostics["maximum_compiled_sigma_d"] < 0.012
    assert diagnostics["amplitude_multiplier"] == 1.0


def test_density_compiled_independent_nps_is_repeatable_ratio_safe_and_nontrivial():
    profile, prior = _parents()
    base = np.full((31, 47, 3), (0.2, 0.4, 0.8), np.float32)
    first, diagnostics = apply_density_compiled_independent_nps(
        base, profile=profile, prior=prior, seed=31
    )
    second, repeated = apply_density_compiled_independent_nps(
        base, profile=profile, prior=prior, seed=31
    )
    assert np.array_equal(first, second)
    assert diagnostics == repeated
    assert np.all(first >= 0.0) and np.all(first <= 1.0)
    assert np.allclose(first[..., 0] / first[..., 1], 0.5)
    assert np.allclose(first[..., 1] / first[..., 2], 0.5)
    assert diagnostics["bounded_residual_rms"] > 0.0
    assert diagnostics["hard_clipping_used"] == 0.0


def test_p4gw_legacy_output_identity_is_unchanged():
    base = np.full((31, 47, 3), (0.2, 0.4, 0.8), np.float32)
    output, _ = apply_independent_density_nps(base, density_sigma=0.0225, seed=31)
    assert hashlib.sha256(memoryview(output).cast("B")).hexdigest() == (
        "43aff49fa0d602bc86253a608c41e8c1807f0119e8a2aea6b22803c9969a740c"
    )
