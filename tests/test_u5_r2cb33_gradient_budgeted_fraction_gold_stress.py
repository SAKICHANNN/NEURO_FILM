from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.gradient_budgeted_fraction_transport import (
    load_contract,
    select_gradient_budgeted_candidate,
)
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb33_gradient_budgeted_fraction_gold_stress_v1.json"
DECISION = (
    ROOT / "configs/u5_r2cb33_gradient_budgeted_fraction_gold_stress_decision_v1.json"
)


def test_cb33_contract_freezes_global_only_dose() -> None:
    contract = load_contract(CONTRACT)
    assert len(contract["operator"]["dose_grid"]) == 17
    assert contract["operator"]["dose_grid"][0] == 1.0
    assert contract["operator"]["dose_grid"][-1] == 0.0
    assert contract["automatic_gates"]["minimum_gold_median_style_delta_e76"] == 5.0


def test_cb33_selector_uses_highest_passing_global_dose() -> None:
    source = np.zeros((32, 32, 3), dtype=np.float32)
    source[:, 16:] = 0.4
    base = source.copy()
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    zero_luma = np.asarray([1.0, -weights[0] / weights[1], 0.0], dtype=np.float32)
    target = source.copy()
    target[:, 16:] += np.float32(0.6) * zero_luma
    candidate, scale, luma_error, facts = select_gradient_budgeted_candidate(
        source,
        base,
        target,
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.75, 0.5, 0.25, 0.0],
        maximum_gradient_ratio=1.5,
    )
    assert facts["global_dose"] == 0.75
    assert facts["selected_gradient_ratio"] <= 1.5
    assert facts["full_candidate_gradient_ratio"] > 1.5
    assert float(np.max(scale)) <= 0.75
    assert float(np.min(scale)) > 0.749
    assert np.max(np.abs(luma_error)) < 1e-7
    assert np.isfinite(candidate).all()


def test_cb33_selector_is_deterministic() -> None:
    rng = np.random.default_rng(20260811 + 3300)
    source = rng.uniform(0.2, 0.7, size=(19, 23, 3)).astype(np.float32)
    base = source.copy()
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    zero_luma = np.asarray([1.0, -weights[0] / weights[1], 0.0], dtype=np.float32)
    target = np.asarray(source + np.float32(0.1) * zero_luma, dtype=np.float32)
    args = {
        "weights": weights,
        "boundary_epsilon": 1.0 / 65535.0,
        "dose_grid": [1.0, 0.5, 0.0],
        "maximum_gradient_ratio": 1.1,
    }
    first = select_gradient_budgeted_candidate(source, base, target, **args)
    second = select_gradient_budgeted_candidate(
        source.copy(), base.copy(), target.copy(), **args
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert first[3] == second[3]


def test_cb33_visual_decision_binds_exact_reviewed_outputs() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    review = decision["visual_review"]
    assert decision["report_sha256"] == (
        "0cd2c971989e0c857aeac7556df39e1171753d4cf0adc3579eb9c9065519060e"
    )
    assert [row["id"] for row in review["original_resolution_outputs"]] == [
        "01", "05", "08", "09", "11", "18", "21", "29"
    ]
    assert all(len(row["sha256"]) == 64 for row in review["original_resolution_outputs"])
    assert review["confirmed_severe_artifact_count"] == 0
    assert decision["decision"] == (
        "pass_development_gold_stress_open_source_disjoint_confirmation"
    )


def test_direction_evaluator_exposes_candidate_builder_hook() -> None:
    assert "candidate_builder" in evaluate_direction_candidate.__annotations__
