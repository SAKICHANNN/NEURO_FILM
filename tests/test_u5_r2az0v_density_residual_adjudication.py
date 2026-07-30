from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.density_residual_adjudication import (
    adjudicate_density_residual_visual,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (
            ROOT / "configs/u5_r2az0v_density_residual_visual_v1.json"
        ).read_text(encoding="utf-8")
    )


def _adjudicate() -> dict:
    return adjudicate_density_residual_visual(
        root=ROOT,
        config=_config(),
        observations_path=(
            ROOT
            / "configs/"
            "u5_r2az0v_density_residual_visual_observations_v1.json"
        ),
        build_dir=ROOT
        / "outputs/u5_r2az0v_density_residual_visual_v1",
    )


def test_az0v_passes_frozen_two_of_three_round_gate() -> None:
    report = _adjudicate()
    assert [
        row["candidate_preferences"] for row in report["rounds"]
    ] == [6, 7, 5]
    assert report["passing_rounds"] == 2
    assert report["confirmed_severe_artifact_count"] == 0
    assert report["blind_gate_passed"] is True
    assert report["production_integration_opened"] is False
    decision = json.loads(
        (
            ROOT
            / "configs/"
            "u5_r2az0v_density_residual_visual_decision_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["visual_adjudication"]["stable_evidence_id"] == report[
        "stable_evidence_id"
    ]
    assert decision["production_integration_opened"] is False
    assert decision["next_leaf"].startswith("U5.R2AZ1")


def test_az0v_rejects_non_blind_observation_claim(tmp_path: Path) -> None:
    source = (
        ROOT
        / "configs/"
        "u5_r2az0v_density_residual_visual_observations_v1.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["mapping_unread_when_recorded"] = False
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not recorded blind"):
        adjudicate_density_residual_visual(
            root=ROOT,
            config=_config(),
            observations_path=path,
            build_dir=ROOT
            / "outputs/u5_r2az0v_density_residual_visual_v1",
        )
