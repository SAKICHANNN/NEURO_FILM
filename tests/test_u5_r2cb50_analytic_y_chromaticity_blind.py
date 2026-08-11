from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = (
    ROOT / "configs/u5_r2cb50_analytic_y_chromaticity_blind_observations_v1.json"
)


def test_cb50_blind_choices_are_complete_and_mapping_sealed() -> None:
    payload = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2CB50"
    assert payload["mapping_opened"] is False
    assert payload["severe_review"]["confirmed_severe_artifact_count"] == 0
    assert len(payload["source_order"]) == 12
    assert [row["round"] for row in payload["rounds"]] == [1, 2, 3]
    assert all(len(row["choices"]) == 12 for row in payload["rounds"])
    assert all(
        choice in {"A", "B"} for row in payload["rounds"] for choice in row["choices"]
    )
