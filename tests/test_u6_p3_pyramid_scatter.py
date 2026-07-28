from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.physical_compiled_scatter import P1_SCHEMA, P3_SCHEMA, load_json
from src.eval.physical_pyramid_scatter import (
    PYRAMID_SCHEMA,
    evaluate_pyramid_scatter,
)
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_pyramid_scatter,
    compile_pyramid_scatter_profile,
)
from src.film_physics.reference_scatter import profile_from_contract


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "configs" / "u6_p1_reference_scatter_simulator_v1.json"
P3 = ROOT / "configs" / "u6_p3_compiled_scatter_challenger_v1.json"
PYRAMID = ROOT / "configs" / "u6_p3_pyramid_scatter_challenger_v1.json"


def test_compiler_keeps_near_direct_and_far_pyramid() -> None:
    reference = profile_from_contract(load_json(P1, P1_SCHEMA))
    profile = compile_pyramid_scatter_profile(
        reference, target_coarse_sigma_pixels=6.0, minimum_pyramid_factor=2
    )
    assert profile.components[0].uses_pyramid is False
    assert profile.components[0].pyramid_factor == 1
    assert profile.components[1].uses_pyramid is True
    assert profile.components[1].pyramid_factor == 5


def test_pyramid_repeat_is_exact() -> None:
    reference = profile_from_contract(load_json(P1, P1_SCHEMA))
    profile = compile_pyramid_scatter_profile(
        reference, target_coarse_sigma_pixels=6.0, minimum_pyramid_factor=2
    )
    values = np.random.default_rng(9).random((79, 83, 3), dtype=np.float32)
    source = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(8.0),
    )
    first = apply_pyramid_scatter(source, profile).values
    second = apply_pyramid_scatter(source, profile).values
    assert np.array_equal(first, second)
    assert np.min(first) >= 0.0


def test_frozen_pyramid_report_is_repeatable() -> None:
    contracts = (
        load_json(P1, P1_SCHEMA),
        load_json(P3, P3_SCHEMA),
        load_json(PYRAMID, PYRAMID_SCHEMA),
    )
    first = evaluate_pyramid_scatter(*contracts)
    second = evaluate_pyramid_scatter(*contracts)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["decisions"]["edge_reference"] is False
    assert all(
        value
        for name, value in first["decisions"].items()
        if name != "edge_reference"
    )
