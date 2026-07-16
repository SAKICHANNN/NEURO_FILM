from __future__ import annotations

import numpy as np

from src.real_film.stock_identifiability import (
    evaluate_structural_support,
    permutation_test,
    roll_balanced_loo,
)


def test_structural_gate_closes_under_supported_stock() -> None:
    rows = []
    for stock, roll_counts in {
        "a": {"a1": 2, "a2": 2, "a3": 2},
        "b": {"b1": 2, "b2": 1, "b3": 2},
    }.items():
        for roll, count in roll_counts.items():
            for index in range(count):
                rows.append({
                    "lane": "negative_preview",
                    "film_stock_id": stock,
                    "roll_id": roll,
                    "content_cell": f"cell{index % 2}",
                })
    gate = evaluate_structural_support(rows, {
        "minimum_rolls_per_stock": 3,
        "minimum_frames_each_roll": 2,
        "minimum_supported_content_cells_per_stock": 2,
        "minimum_frames_per_supported_content_cell": 2,
        "minimum_shared_supported_content_cells_per_stock_pair": 2,
        "minimum_stocks_in_comparable_clique": 2,
    })
    assert gate["support_by_stock"]["b"]["passed"] is False
    assert "under_supported_roll" in gate["support_by_stock"]["b"]["failure_reasons"]
    assert gate["passed"] is False


def test_roll_balanced_loo_uses_one_vote_per_roll() -> None:
    roll_ids = ["a1"] * 5 + ["a2"] * 2 + ["a3"] * 2 + ["b1"] * 4 + ["b2"] * 2 + ["b3"] * 2
    labels = ["a"] * 9 + ["b"] * 8
    features = np.asarray([[0.0]] * 5 + [[0.1]] * 2 + [[0.2]] * 2 + [[10.0]] * 4 + [[10.1]] * 2 + [[10.2]] * 2)
    result = roll_balanced_loo(features, roll_ids, labels)
    assert result["rolls"] == 6
    assert result["accuracy"] == 1.0
    assert result["per_stock_roll_recall"] == {"a": 1.0, "b": 1.0}


def test_permutation_test_is_deterministic() -> None:
    roll_ids = [roll for roll in ("a1", "a2", "a3", "b1", "b2", "b3") for _ in range(2)]
    labels = ["a"] * 6 + ["b"] * 6
    features = np.asarray([[0.0], [0.1], [0.2], [0.1], [0.3], [0.2], [9.8], [10.0], [10.1], [9.9], [10.2], [10.0]])
    first = permutation_test(features, roll_ids, labels, permutations=31, seed=7)
    second = permutation_test(features, roll_ids, labels, permutations=31, seed=7)
    assert first == second
    assert first["observed_accuracy"] == 1.0
