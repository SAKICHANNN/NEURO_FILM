from __future__ import annotations

import json
from pathlib import Path

from src.eval.rawpixls_confirmation_preflight import validate_contract
from src.eval.fresh_strength_adjudication import adjudicate_fresh_strength


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u6_p8bn_fresh_strength_population_v1.json"
)
ADJUDICATION = (
    ROOT / "configs/u6_p8bn_fresh_strength_adjudication_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bn_fresh_strength_population_decision_v1.json"
)


def test_p8bn_contract_is_fresh_cc0_and_strength_frozen() -> None:
    config = json.loads(CONFIG.read_text())
    validate_contract(ROOT, config)
    rows = config["candidates"]
    assert len(rows) == 8
    assert len({row["make"] for row in rows}) == 8
    assert len({row["repository_id"] for row in rows}) == 8
    assert all(row["url"].startswith(
        "https://raw.pixls.us/getfile.php/"
    ) for row in rows)
    assert config["comparison"]["strengths"] == [0.8, 1.0]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]


def test_p8bn_adjudication_retains_full_strength() -> None:
    config = json.loads(ADJUDICATION.read_text())
    report = adjudicate_fresh_strength(root=ROOT, config=config)
    assert report["automatic_gate_pass"]
    assert report["severe_artifact_gate_pass"]
    assert report["contact_sheet_confirmed_new_severe_count"] == 0
    assert report["full_resolution_confirmed_new_severe_count"] == 0
    assert report["strength_round_wins"] == {
        "0.8": 1,
        "1.0": 2,
        "tie": 0,
    }
    assert not report["preference_gate_pass"]
    assert report["decision"] == (
        "retain_1.0_and_close_0.8_global_preference_challenge"
    )
    assert not report["production_default_changed"]


def test_p8bn_scene_vote_totals_are_three_each() -> None:
    config = json.loads(ADJUDICATION.read_text())
    report = adjudicate_fresh_strength(root=ROOT, config=config)
    assert all(
        counts["0.8"] + counts["1.0"] == 3
        for counts in report["per_scene_vote_counts"].values()
    )


def test_p8bn_decision_closes_reduced_global_default() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["comparison"]["strength_0.8_round_wins"] == 1
    assert decision["comparison"]["strength_1.0_round_wins"] == 2
    assert not decision["comparison"]["preference_gate_pass"]
    assert decision["result"]["content_router_opened"] is False
    assert decision["result"]["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8BO")
