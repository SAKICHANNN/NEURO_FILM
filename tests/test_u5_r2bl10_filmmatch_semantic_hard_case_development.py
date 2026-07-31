from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.filmmatch_semantic_hard_case import (
    evaluate_selector,
    hard_top1_predictions,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "configs/u5_r2bl10_filmmatch_semantic_hard_case_development_v1.json").read_text(encoding="utf-8"))


def test_contract_is_frozen_and_parent_bound() -> None:
    decision, rows = validate_contract(ROOT, _config())
    assert decision["decision"] == "retain_ao6_incumbent_close_bl5_preference_promotion"
    assert len(rows) == 17
    assert len({row["make"] for row in rows}) == 17


def test_hard_top1_excludes_self_and_uses_nearest_label() -> None:
    features = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    labels = np.asarray([True, True, False])
    predictions, rows = hard_top1_predictions(features, labels, ["a", "b", "c"])
    assert predictions.tolist() == [True, True, True]
    assert [row["nearest_source_id"] for row in rows] == ["b", "a", "b"]
    assert all(row["source_id"] != row["nearest_source_id"] for row in rows)


def test_selector_passes_when_semantic_pairs_recover_oracle() -> None:
    features = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    choices = np.asarray([3, 3, 0, 0])
    config = _config()
    config["automatic_gate"].update(
        minimum_selector_choice_gain_over_global_ao6=4,
        minimum_selector_correct_source_gain_over_global_ao6=2,
        minimum_candidate_selected_sources=2,
        maximum_false_candidate_selections=0,
        minimum_choice_gain_over_negative_control_p95=-100,
    )
    result = evaluate_selector(features=features, source_ids=["a", "b", "c", "d"], candidate_choices=choices, config=config)
    assert result["selector_choice_gain"] == 6
    assert result["selector_correct_sources"] == 4
    assert result["passed"] is True


def test_hard_top1_rejects_nonfinite_features() -> None:
    with pytest.raises(ValueError, match="finite"):
        hard_top1_predictions(np.asarray([[1.0], [np.nan], [0.5]]), np.asarray([True, False, False]), ["a", "b", "c"])


def test_formal_report_is_exact_and_closes_selector() -> None:
    output = ROOT / "outputs/u5_r2bl10_filmmatch_semantic_hard_case_development_v1"
    assert (output / "run_a.json").read_bytes() == (output / "run_b.json").read_bytes()
    report = json.loads((output / "run_a.json").read_text(encoding="utf-8"))
    assert report["selector_choice_gain"] == 3
    assert report["selector_correct_source_gain"] == 1
    assert report["candidate_selected_sources"] == 3
    assert report["false_candidate_selections"] == 1
    assert report["negative_control_choice_gain_p95"] == 3.0
    assert report["passed"] is False
    assert report["decision"] == "close_semantic_hard_case_retain_ao6"
