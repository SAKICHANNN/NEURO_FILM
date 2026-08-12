import json
from pathlib import Path

import numpy as np

from src.film_physics.density_lod_residual import apply_density_lod_residual

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gu_compound_poisson_density_photographic_development_v1.json"


def test_p4gu_freezes_compound_poisson_tail_before_execution():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert (
        contract["candidate"]["residual_projection"]
        == "compound-poisson-density-multiplicative"
    )
    assert contract["candidate"]["tail_sigma"] == 3.0
    assert contract["source"]["freshness"].startswith("consumed P4GR-P4GT")
    assert contract["execution"]["same_cohort_rescue_allowed"] is False


def test_compound_poisson_tail_is_smooth_bounded_and_nonzero():
    base = np.full((9, 11, 3), (0.2, 0.4, 0.8), dtype=np.float32)
    reference = np.full_like(base, 0.5)
    physical = reference.copy()
    physical[4, 5] = 1.0e-4
    unbounded, _ = apply_density_lod_residual(base, physical, reference)
    bounded, diagnostics = apply_density_lod_residual(
        base, physical, reference, finite_tail_density=0.12
    )
    assert np.max(np.abs(bounded - base)) < np.max(np.abs(unbounded - base))
    assert np.max(np.abs(bounded - base)) > 0.0
    assert diagnostics["finite_tail_density"] == 0.12
    assert np.allclose(bounded[..., 0] / bounded[..., 1], 0.5)
    assert np.allclose(bounded[..., 1] / bounded[..., 2], 0.5)


def test_compound_poisson_tail_rejects_invalid_sigma():
    values = np.full((3, 3, 3), 0.5, dtype=np.float32)
    for invalid in (0.0, -1.0, np.inf):
        try:
            apply_density_lod_residual(
                values, values, values, finite_tail_density=invalid
            )
        except ValueError as error:
            assert "finite-tail" in str(error)
        else:
            raise AssertionError("invalid finite-tail density was accepted")
