from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.crystal_light_depletion import (
    CrystalLightDepletionEvaluationError,
    _validate_contract,
    evaluate_crystal_light_depletion,
)
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.crystal_light_depletion import (
    CrystalLightDepletionProfile,
    build_crystal_footprint_map,
    render_crystal_light_depletion,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cc_crystal_light_depletion_v1.json"


def _profile() -> CrystalLightDepletionProfile:
    return CrystalLightDepletionProfile(
        profile_id="test-crystals",
        layer_count_per_channel=4,
        cell_pitch_pixels=8,
        activation_probability=0.85,
        radius_min_pixels=2.25,
        radius_max_pixels=3.75,
        centre_jitter_fraction=0.5,
        capture_efficiency=0.32,
        channel_seed_bases=(11, 29, 47),
        layer_seed_stride=101,
    )


def _exposure() -> PhysicalDomainArray:
    values = np.full((31, 37, 3), 0.4, dtype=np.float64)
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(1.0),
    )


def test_crystal_geometry_is_bounded_repeatable_and_channel_distinct() -> None:
    profile = _profile()
    first = build_crystal_footprint_map(
        (31, 37), profile, channel_index=0, layer_index=0
    )
    repeated = build_crystal_footprint_map(
        (31, 37), profile, channel_index=0, layer_index=0
    )
    other = build_crystal_footprint_map(
        (31, 37), profile, channel_index=1, layer_index=0
    )
    assert np.array_equal(first.labels, repeated.labels)
    assert first.geometry_id == repeated.geometry_id
    assert first.geometry_id != other.geometry_id
    assert np.min(first.labels) >= -1
    assert 0.0 < first.coverage_fraction < 1.0


def test_capture_conserves_positive_layer_exposure() -> None:
    exposure = _exposure()
    result = render_crystal_light_depletion(exposure, _profile())
    assert (
        np.max(
            np.abs(
                result.captured_exposure.values
                + result.remaining_exposure.values
                - exposure.values
            )
        )
        <= 2e-15
    )
    assert np.all(result.captured_exposure.values >= 0.0)
    assert np.all(result.remaining_exposure.values > 0.0)
    assert np.all(result.captured_exposure.values <= exposure.values)


def test_contract_rejects_capture_gate_drift() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["evaluation"]["minimum_constant_total_capture_fraction"] = 0.6
    with pytest.raises(CrystalLightDepletionEvaluationError):
        _validate_contract(contract)


def test_formal_evaluation_is_exact_and_gate_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_crystal_light_depletion(contract, ROOT)
    second = evaluate_crystal_light_depletion(contract, ROOT)
    assert first == second
    assert set(first["checks"]) == {
        "energy_conservation",
        "physical_domain",
        "constant_capture",
        "layer_depletion",
        "footprint_detail_loss",
        "no_depletion_control_violates_energy",
        "layer_order_is_material",
        "density_transmittance_roundtrip",
        "distinct_channel_crystal_maps",
        "exact_repeat",
        "finite",
    }
