from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.fivek_monotone_curve_blind_review import (
    FiveKMonotoneCurveBlindError,
    score_judgments,
    select_rows,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay7_monotone_curve_blind_product_value_v1.json"
)


def test_contract_binds_confirmed_parent_and_hidden_labels() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["report"]["automatic_pass"] is True
    assert config["comparison"]["candidate_labels_hidden"] is True
    assert config["production_integration_allowed"] is False


def test_selection_is_fixed_across_four_benefit_strata() -> None:
    rows = [
        {
            "pair_id": f"pair-{index:02d}",
            "ridge": {"neutral_rmse": float(index)},
            "global": {"neutral_rmse": 0.0},
        }
        for index in range(64)
    ]
    selected = select_rows(rows)
    assert len(selected) == 16
    assert [row["pair_id"] for row in selected] == [
        f"pair-{index:02d}"
        for index in (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61)
    ]


def _write_scoring_fixture(tmp_path: Path) -> dict:
    config = {
        "experiment_id": "blind-test",
        "selection": {
            "rows": 2,
            "round_seeds": [11, 12, 13],
            "same_rows_each_round": True,
        },
        "evaluation": {
            "minimum_ridge_preferences_per_round": 2,
            "maximum_ties_per_round": 0,
            "minimum_passing_rounds": 2,
            "maximum_confirmed_severe_failures": 0,
        },
        "assessment_label": "autonomous blinded visual target-match evidence",
        "claim_ceiling": "test-only",
    }
    private = {"experiment_id": "blind-test", "rounds": []}
    choices = [("A", "B"), ("A", "B"), ("A", "A")]
    for round_number, round_choices in enumerate(choices, start=1):
        items = [
            {
                "blind_id": f"R{round_number}-01",
                "pair_id": "pair-a",
                "ridge_slot": "A",
                "global_slot": "B",
            },
            {
                "blind_id": f"R{round_number}-02",
                "pair_id": "pair-b",
                "ridge_slot": "B",
                "global_slot": "A",
            },
        ]
        private["rounds"].append(
            {
                "round": round_number,
                "seed": 10 + round_number,
                "items": items,
            }
        )
        (tmp_path / f"round_{round_number}_judgment.json").write_text(
            json.dumps(
                {
                    "round": round_number,
                    "assessment_label": (
                        "autonomous blinded visual target-match evidence"
                    ),
                    "items": [
                        {
                            "blind_id": f"R{round_number}-01",
                            "choice": round_choices[0],
                            "severe": False,
                        },
                        {
                            "blind_id": f"R{round_number}-02",
                            "choice": round_choices[1],
                            "severe": False,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    key_path = tmp_path / "private_key.json"
    key_path.write_text(json.dumps(private), encoding="utf-8")
    (tmp_path / "pack.json").write_text(
        json.dumps(
            {
                "experiment_id": "blind-test",
                "selected_pair_count": 2,
                "private_key": "private_key.json",
                "private_key_sha256": hashlib.sha256(
                    key_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return config


def test_score_judgments_applies_frozen_round_gate(
    tmp_path: Path,
) -> None:
    config = _write_scoring_fixture(tmp_path)
    report = score_judgments(config=config, pack_dir=tmp_path)
    assert report["automatic_pass"] is True
    assert report["passing_rounds"] == 2
    assert [
        round_result["ridge_preferences"]
        for round_result in report["rounds"]
    ] == [2, 2, 1]


def test_score_judgments_rejects_duplicate_rows(
    tmp_path: Path,
) -> None:
    config = _write_scoring_fixture(tmp_path)
    judgment_path = tmp_path / "round_1_judgment.json"
    judgment = json.loads(judgment_path.read_text(encoding="utf-8"))
    judgment["items"][1]["blind_id"] = judgment["items"][0]["blind_id"]
    judgment_path.write_text(json.dumps(judgment), encoding="utf-8")
    with pytest.raises(
        FiveKMonotoneCurveBlindError, match="coverage drift"
    ):
        score_judgments(config=config, pack_dir=tmp_path)
