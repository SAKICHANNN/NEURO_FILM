from __future__ import annotations

import json
from pathlib import Path

from src.eval.factorization_adjudication import adjudicate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_blind_adjudication_closes_preference() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2ba1v_hue_value_residual_adjudication_v1.json"
        ).read_text(encoding="utf-8")
    )
    report = adjudicate(root=ROOT, config=config)
    assert [row["candidate_preferences"] for row in report["rounds"]] == [
        5,
        4,
        3,
    ]
    assert report["passing_rounds"] == 0
    assert report["confirmed_severe_artifact_count"] == 0
    assert report["visual_pass"] is False
