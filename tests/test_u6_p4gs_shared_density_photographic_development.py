import json
from pathlib import Path

import numpy as np

from src.film_physics.bounded_common_density_residual import (
    apply_bounded_common_density_residual,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gs_shared_density_photographic_development_v1.json"


def test_p4gs_freezes_shared_density_mechanism_before_execution():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert (
        contract["candidate"]["residual_projection"] == "shared-density-multiplicative"
    )
    assert contract["source"]["freshness"].startswith("consumed P4GR")
    assert contract["execution"]["same_cohort_rescue_allowed"] is False
    assert len(contract["mechanism_sources"]) == 2


def test_common_density_residual_preserves_rgb_ratios_and_bounds():
    base = np.array([[[0.2, 0.4, 0.8], [0.5, 0.25, 0.125]]], dtype=np.float32)
    reference = np.full_like(base, 0.5)
    physical = np.array([[[0.25, 0.4, 0.5], [0.8, 0.7, 0.6]]], dtype=np.float32)
    output, diagnostics = apply_bounded_common_density_residual(
        base, physical, reference
    )
    assert np.all(output >= 0.0) and np.all(output <= 1.0)
    assert np.allclose(output[..., 0] / output[..., 1], base[..., 0] / base[..., 1])
    assert np.allclose(output[..., 1] / output[..., 2], base[..., 1] / base[..., 2])
    assert diagnostics["removed_chromatic_density_rms"] > 0.0
    assert diagnostics["hard_clipping_used"] == 0.0


def test_common_density_residual_rejects_zero_scan_values():
    base = np.full((1, 1, 3), 0.5, dtype=np.float32)
    reference = np.full_like(base, 0.5)
    physical = reference.copy()
    physical[0, 0, 0] = 0.0
    try:
        apply_bounded_common_density_residual(base, physical, reference)
    except ValueError as error:
        assert "invalid common-density" in str(error)
    else:
        raise AssertionError("zero scan value was accepted")
