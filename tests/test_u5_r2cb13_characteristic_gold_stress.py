from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.characteristic_gold_stress import (
    CharacteristicGoldStressError,
    _encode_png,
    adjudicate_files,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cb13_contract_is_frozen_partial_gold_scope() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb13_characteristic_gold_stress_v1.json"
    )
    assert contract["population"]["required_unavailable_ids"] == ["FS_FACE_01"]
    assert contract["population"]["expected_available_source_count"] == 40
    assert "not the complete gold set" in contract["claim_ceiling"]


def test_cb13_png_encoding_is_exact_and_bounded() -> None:
    source = np.linspace(0.0, 1.0, 3 * 9 * 11, dtype=np.float32).reshape(9, 11, 3)
    first, pixels = _encode_png(source)
    second, replay = _encode_png(source.copy())
    assert first == second
    assert np.array_equal(pixels, replay)
    assert pixels.dtype == np.uint8


def test_cb13_adjudicates_exact_partial_evidence() -> None:
    base = ROOT / "outputs/experiments/u5_r2cb13_characteristic_gold_stress_v1"
    result = adjudicate_files(
        config_path=ROOT / "configs/u5_r2cb13_characteristic_gold_stress_v1.json",
        report_paths=[base / "report_a.json", base / "report_b.json"],
        review_path=ROOT
        / "configs/u5_r2cb13_characteristic_gold_stress_visual_review_v1.json",
        adjudicator_software_commit="0" * 40,
    )
    assert result["partial_available_cohort_pass"] is True
    assert result["complete_gold_set_pass"] is False
    assert result["measurements"]["maximum_new_hard_boundary_fraction"] == 0.0
    assert result["confirmed_severe_artifact_count"] == 0


def test_cb13_rejects_review_identity_drift(tmp_path: Path) -> None:
    base = ROOT / "outputs/experiments/u5_r2cb13_characteristic_gold_stress_v1"
    source = ROOT / "configs/u5_r2cb13_characteristic_gold_stress_visual_review_v1.json"
    review = json.loads(source.read_text(encoding="utf-8"))
    review["original_resolution_outputs"][0]["sha256"] = "0" * 64
    mutated = tmp_path / "review.json"
    mutated.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(CharacteristicGoldStressError):
        adjudicate_files(
            config_path=ROOT / "configs/u5_r2cb13_characteristic_gold_stress_v1.json",
            report_paths=[base / "report_a.json", base / "report_b.json"],
            review_path=mutated,
            adjudicator_software_commit="0" * 40,
        )


def test_cb13_decision_matches_exact_evidence() -> None:
    base = ROOT / "outputs/experiments/u5_r2cb13_characteristic_gold_stress_v1"
    decision = json.loads(
        (
            ROOT / "configs/u5_r2cb13_characteristic_gold_stress_decision_v1.json"
        ).read_text(encoding="utf-8")
    )
    actual = adjudicate_files(
        config_path=ROOT / "configs/u5_r2cb13_characteristic_gold_stress_v1.json",
        report_paths=[base / "report_a.json", base / "report_b.json"],
        review_path=ROOT
        / "configs/u5_r2cb13_characteristic_gold_stress_visual_review_v1.json",
        adjudicator_software_commit=decision["adjudicator_software_commit"],
    )
    assert actual == decision
