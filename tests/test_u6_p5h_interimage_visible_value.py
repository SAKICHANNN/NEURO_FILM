from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_interimage_visible_value import (
    _encoded_srgb8,
    _visibility_metrics,
    load_contract,
    score_blind_choices,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p5h_interimage_visible_value_v1.json"


def test_contract_freezes_three_blind_rounds_and_tie_rule() -> None:
    contract = load_contract(CONTRACT)
    assert contract["blind_visual_protocol"]["round_seeds"] == [
        62081,
        62082,
        62083,
    ]
    assert "tie" in contract["blind_visual_protocol"]["tie_rule"]


def test_encoded_srgb8_is_exact_and_bounded() -> None:
    values = np.array([[[0.0, 0.18, 1.0]]], dtype=np.float64)
    first = _encoded_srgb8(values)
    second = _encoded_srgb8(values)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.tolist()[0][0][0] == 0
    assert first.tolist()[0][0][2] == 255


def test_visibility_metrics_detect_edge_concentrated_change() -> None:
    source = np.full((32, 64, 3), 0.2, dtype=np.float64)
    source[:, 32:, 0] = 0.8
    baseline = source.copy()
    candidate = source.copy()
    candidate[:, 31, 0] -= 0.01
    candidate[:, 32, 0] += 0.01
    metrics = _visibility_metrics(source, baseline, candidate)
    assert metrics["changed_pixel_fraction"] > 0.0
    assert metrics["strong_edge_opponent_gain_fraction"] > 0.0
    assert metrics["strong_edge_to_flat_changed_ratio"] > 1.0


def test_blind_scoring_counts_tie_against_candidate() -> None:
    contract = load_contract(CONTRACT)
    key = {
        "rounds": [
            {
                "round": round_index,
                "rows": [
                    {
                        "display_row": row + 1,
                        "id": str(row),
                        "candidate_side": "A",
                    }
                    for row in range(9)
                ],
            }
            for round_index in range(1, 4)
        ]
    }
    passing = {round_index: ["A"] * 6 + ["tie"] * 3 for round_index in range(1, 4)}
    assert score_blind_choices(contract, key, passing)["visual_pass"] is True
    failing = {**passing, 2: ["A"] * 5 + ["tie"] * 4}
    assert score_blind_choices(contract, key, failing)["visual_pass"] is False


def test_blind_scoring_rejects_wrong_choice_count() -> None:
    contract = load_contract(CONTRACT)
    key = {"rounds": [{"round": 1, "rows": [{"candidate_side": "A"}] * 9}]}
    with pytest.raises(ValueError, match="count"):
        score_blind_choices(contract, key, {1: ["A"] * 8})
