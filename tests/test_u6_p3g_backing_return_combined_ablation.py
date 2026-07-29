from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_backing_return_combined_ablation import (
    _compile_profiles,
    _diagnostic_map,
    _variants,
    load_contract,
)
from src.eval.sensitometry_primitive import build_operator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p3g_backing_return_combined_ablation_v1.json"
P1 = ROOT / "configs" / "u6_p1_reference_scatter_simulator_v1.json"
P3D = ROOT / "configs" / "u6_p3d_backing_return_reference_v1.json"
SENSITOMETRY = ROOT / "configs" / "u2_2a_sensitometry_primitive_v1.json"


def test_contract_freezes_four_variants_and_forbids_double_counting() -> None:
    contract = load_contract(CONTRACT)
    assert len(contract["fixed_chain"]["variants"]) == 4
    assert (
        contract["fixed_chain"]["candidate"]
        == "P1-forward-only-then-P3E-additive-return"
    )
    assert contract["production_integration_allowed"] is False


def test_split_profiles_are_scale_aligned_and_distinct() -> None:
    legacy, forward, backing = _compile_profiles(
        json.loads(P1.read_text("utf-8")),
        json.loads(P3D.read_text("utf-8")),
    )
    assert legacy.pixel_pitch_um == forward.pixel_pitch_um == backing.pixel_pitch_um
    assert len(legacy.kernels) == 2
    assert len(forward.kernels) == 1
    assert len(backing.kernels) == 2


def test_small_fixed_chain_is_finite_bounded_and_mechanistically_distinct() -> None:
    legacy, forward, backing = _compile_profiles(
        json.loads(P1.read_text("utf-8")),
        json.loads(P3D.read_text("utf-8")),
    )
    operator = build_operator(json.loads(SENSITOMETRY.read_text("utf-8")))
    source = np.random.default_rng(119).random((37, 41, 3), dtype=np.float32)
    variants = _variants(
        source,
        legacy=legacy,
        forward=forward,
        backing=backing,
        operator=operator,
    )
    assert all(
        np.all(np.isfinite(value))
        and np.all(value > 0.0)
        and np.all(value <= 1.0)
        for value in variants.values()
    )
    assert not np.array_equal(
        variants["split_candidate"], variants["legacy_full"]
    )
    assert not np.array_equal(
        variants["split_candidate"], variants["forbidden_double"]
    )


def test_diagnostic_map_is_finite_and_bounded() -> None:
    transmittance = np.linspace(0.01, 1.0, 12, dtype=np.float64).reshape(2, 2, 3)
    mapped = _diagnostic_map(transmittance)
    assert np.all(np.isfinite(mapped))
    assert np.all((mapped >= 0.0) & (mapped <= 1.0))
