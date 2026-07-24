from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.hard_routing_visual_audit import (
    _fit_image,
    adjudicate_scores,
    blind_assignment,
)


def test_blind_assignment_is_stable_and_round_specific() -> None:
    first = [
        blind_assignment(20260723, "full_fit", sample_id)
        for sample_id in ("06", "11", "12", "18", "20", "23", "24", "28")
    ]
    second = [
        blind_assignment(20260723, "full_fit", sample_id)
        for sample_id in ("06", "11", "12", "18", "20", "23", "24", "28")
    ]
    other_round = [
        blind_assignment(20260723, "detail_crop", sample_id)
        for sample_id in ("06", "11", "12", "18", "20", "23", "24", "28")
    ]
    assert first == second
    assert first != other_round


def test_fit_image_has_exact_canvas_for_full_and_crop(tmp_path: Path) -> None:
    array = np.zeros((100, 160, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(160, dtype=np.uint8)[None, :]
    path = tmp_path / "input.png"
    Image.fromarray(array, mode="RGB").save(path)
    full = _fit_image(path, (80, 80), False)
    crop = _fit_image(path, (80, 80), True)
    assert full.size == (80, 80)
    assert crop.size == (80, 80)
    assert np.asarray(full).shape == (80, 80, 3)
    assert not np.array_equal(np.asarray(full), np.asarray(crop))


def test_adjudication_applies_every_frozen_gate() -> None:
    ids = [f"{index:02d}" for index in range(10)]
    rounds = []
    key = []
    for round_id in ("one", "two", "three"):
        rows = []
        for index, sample_id in enumerate(ids):
            a_is_routed = index % 2 == 0
            key.append(
                {
                    "round_id": round_id,
                    "sample_id": sample_id,
                    "a_is_routed": a_is_routed,
                }
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "a_severe": False,
                    "b_severe": False,
                    "overall": (
                        "A" if a_is_routed and index < 7 else
                        "B" if not a_is_routed and index < 7 else
                        "B" if a_is_routed else "A"
                    ),
                }
            )
        rounds.append({"round_id": round_id, "rows": rows})
    severe = {
        "records": [
            {
                "sample_id": f"{index:02d}",
                "routed_severe": False,
                "global_severe": False,
            }
            for index in range(41)
        ]
    }
    result = adjudicate_scores(
        {
            "status": "frozen_before_blind_key_open",
            "rounds": rounds,
        },
        key,
        severe,
        {
            "maximum_routed_severe_failures": 0,
            "maximum_new_intervention_severe_failures": 0,
            "minimum_routed_win_plus_tie_per_round": 7,
            "minimum_positive_score_rounds": 2,
        },
    )
    assert result["decision"] == (
        "retain_exact_hard_1nn_as_a0_research_challenger"
    )
    assert all(row["routed_wins"] == 7 for row in result["rounds"])
