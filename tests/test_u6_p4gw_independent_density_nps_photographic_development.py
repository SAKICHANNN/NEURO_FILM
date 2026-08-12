import json
from pathlib import Path

import numpy as np

from src.film_physics.independent_density_nps import (
    apply_independent_density_nps,
    synthesize_independent_density_nps,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p4gw_contract_reuses_safe_density_without_claim_rescue():
    c = json.loads(
        (
            ROOT
            / "configs/u6_p4gw_independent_density_nps_photographic_development_v1.json"
        ).read_text()
    )
    assert c["candidate"]["density_sigma"] == 0.0225
    assert c["parents"]["safe_density_primitive"]["required_visual_value_pass"] is False
    assert c["execution"]["same_cohort_rescue_allowed"] is False


def test_independent_density_nps_is_exact_zero_dc_unit_std_and_repeatable():
    a, da = synthesize_independent_density_nps((31, 47), seed=19)
    b, db = synthesize_independent_density_nps((31, 47), seed=19)
    assert np.array_equal(a, b)
    assert abs(da["field_mean"]) < 1e-15
    assert abs(da["field_std"] - 1) < 1e-15
    assert da == db


def test_independent_density_nps_preserves_ratios_and_cube():
    base = np.full((31, 47, 3), (0.2, 0.4, 0.8), np.float32)
    out, d = apply_independent_density_nps(base, density_sigma=0.0225, seed=31)
    assert np.all(out >= 0) & np.all(out <= 1)
    assert np.allclose(out[..., 0] / out[..., 1], 0.5)
    assert np.allclose(out[..., 1] / out[..., 2], 0.5)
    assert d["bounded_residual_rms"] > 0
    assert d["hard_clipping_used"] == 0
