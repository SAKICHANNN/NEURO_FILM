from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_backing_return_combined_ablation import (
    _compile_profiles,
)
from src.eval.scene_linear_backing_return import (
    _resize_scene_linear,
    _variants_fft,
    load_contract,
    validate_contract,
)
from src.eval.sensitometry_primitive import build_operator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p3l_scene_linear_backing_return_v1.json"
)
DECISION = (
    ROOT / "configs/u6_p3l_scene_linear_backing_return_decision_v1.json"
)


def test_contract_is_scene_linear_data_gated_and_non_promotional() -> None:
    contract = load_contract(CONTRACT)
    assert contract["input"]["decode"].startswith("existing LibRaw")
    assert contract["fixed_chain"]["profile_fitting"] is False
    assert contract["fixed_chain"]["parameter_selection"] is False
    assert contract["production_integration_allowed"] is False
    assert "reopen P3G" in contract["forbidden_actions"][4]


def test_contract_reuses_exact_nine_source_gate() -> None:
    contract = load_contract(CONTRACT)
    manifest, _, _, _, _ = validate_contract(ROOT, contract)
    assert len(manifest) == 9
    assert len({row["make"] for row in manifest}) == 9


def test_scene_linear_resize_is_bounded_and_deterministic() -> None:
    source = np.random.default_rng(31).random(
        (47, 83, 3), dtype=np.float32
    )
    first = _resize_scene_linear(source, 41)
    second = _resize_scene_linear(source, 41)
    assert first.shape == (23, 41, 3)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_fft_fixed_chain_is_finite_and_mechanistically_distinct() -> None:
    contract = load_contract(CONTRACT)
    _, _, p1, p3d, sensitometry = validate_contract(ROOT, contract)
    legacy, forward, backing = _compile_profiles(p1, p3d)
    operator = build_operator(sensitometry)
    source = np.random.default_rng(53).random(
        (61, 67, 3), dtype=np.float32
    )
    variants = _variants_fft(
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


def test_decision_closes_without_visual_or_threshold_rescue() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    run_a = ROOT / decision["repeat_evidence"]["run_a"]
    run_b = ROOT / decision["repeat_evidence"]["run_b"]
    assert run_a.read_bytes() == run_b.read_bytes()
    assert decision["automatic_gate"]["passed"] is False
    assert decision["automatic_gate"]["visual_review_allowed"] is False
    assert decision["metrics"]["maximum_candidate_vs_no_spatial_abs"] > (
        decision["metrics"]["maximum_candidate_vs_no_spatial_abs_gate"]
    )
    assert decision["next_leaf"].startswith("U6.P3M read-only")
