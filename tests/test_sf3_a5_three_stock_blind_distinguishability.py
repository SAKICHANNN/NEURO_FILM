from __future__ import annotations

import pytest

from src.real_film.three_stock_blind_distinguishability import (
    ThreeStockBlindError,
    adjudicate,
    build_mapping,
)

STOCKS = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
SCENES = ["scene-1", "scene-2", "scene-3", "scene-4"]


def test_exact_three_way_assignments_pass() -> None:
    mapping = build_mapping(SCENES, STOCKS, seed="hidden-seed")
    observations = [
        {
            "round": row["round"],
            "scene_id": row["scene_id"],
            "label_to_stock": row["label_to_stock"],
        }
        for row in mapping
    ]
    report = adjudicate(mapping, observations)
    assert report["automatic_pass"] is True
    assert report["overall_assignment_accuracy"] == 1.0


def test_incomplete_and_collapsed_assignments_fail_closed() -> None:
    mapping = build_mapping(SCENES, STOCKS, seed="hidden-seed")
    with pytest.raises(ThreeStockBlindError, match="incomplete"):
        adjudicate(mapping, [])
    observations = []
    for row in mapping:
        labels = sorted(row["label_to_stock"])
        shifted = {
            label: row["label_to_stock"][labels[(index + 1) % 3]]
            for index, label in enumerate(labels)
        }
        observations.append(
            {
                "round": row["round"],
                "scene_id": row["scene_id"],
                "label_to_stock": shifted,
            }
        )
    assert adjudicate(mapping, observations)["automatic_pass"] is False
