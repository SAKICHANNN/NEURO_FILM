from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.photographic_material_stress import (
    bound_material_density,
    render_material_arms,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4u_photographic_material_stress_v1.json"


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p4u_contract_binds_retained_physical_parents() -> None:
    runtime, gauge, report, p4d, p4t = validate_contract(ROOT, _config())
    assert len(runtime.eligible_ids) == 16
    assert report["selected_candidate_arm_id"] == "gauged_spatial_4000"
    assert len(gauge.inverse_neutral_splines) == 3
    assert len(p4d["profiles"]) == 3
    assert p4t["candidate"]["sigma_yx_pixels"] == [0.9, 0.65]


def test_p4u_changes_only_material_kernel_and_repeats_exactly() -> None:
    runtime, gauge, _, p4d, p4t = validate_contract(ROOT, _config())
    source = np.random.default_rng(2026072904).random((33, 37, 3))
    first, first_diagnostics = render_material_arms(
        source, runtime, gauge, p4d, p4t, sampling_dpi=4000
    )
    second, second_diagnostics = render_material_arms(
        source, runtime, gauge, p4d, p4t, sampling_dpi=4000
    )
    assert list(first) == _config()["chain"]["arms"]
    assert first_diagnostics == second_diagnostics
    for arm_id in first:
        np.testing.assert_array_equal(first[arm_id], second[arm_id])
        assert np.all((first[arm_id] >= 0.0) & (first[arm_id] <= 1.0))
    assert not np.array_equal(
        first["p4d_isotropic_shared_seed"],
        first["p4t_anisotropic_shared_seed"],
    )
    for row in first_diagnostics.values():
        assert row["target_minimum"] >= 0.0
        assert row["target_maximum"] <= 2.0
        assert row["raw_material_minimum"] >= 0.0
        assert row["bounded_material_minimum"] >= 0.0
        assert 0.0 <= row["minimum_residual_scale"] <= 1.0


def test_p4u_rejects_target_outside_material_domain() -> None:
    target = np.full((17, 19, 3), 0.5, dtype=np.float64)
    target[0, 0, 0] = 2.1
    material = target.copy()
    try:
        bound_material_density(
            target,
            material,
            lower_rgb=np.zeros(3),
            upper_rgb=np.full(3, 2.0),
        )
    except ValueError as error:
        assert "invalid material density" in str(error)
    else:
        raise AssertionError("out-of-domain density must fail closed")
