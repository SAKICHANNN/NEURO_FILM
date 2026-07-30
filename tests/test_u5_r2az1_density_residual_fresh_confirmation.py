from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.density_residual_fresh_confirmation import validate_contract
from src.eval.density_residual_fresh_visual import (
    build_fresh_visual_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_az1_contract_freezes_exact_candidate_and_fresh_population() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2az1_density_residual_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["status"] == "contract_frozen_before_candidate_render"
    assert config["candidate"]["neutral_strength"] == 0.15
    assert config["candidate"]["opponent_strength"] == 0.35
    assert config["fresh_population"]["required_sample_count"] == 16
    assert config["fresh_population"]["required_camera_make_count"] == 9
    assert (
        config["fresh_population"][
            "required_zero_decoded_sha256_overlap_with_az0_development"
        ]
        is True
    )
    assert config["visual_protocol"][
        "minimum_candidate_preferences_per_passing_round"
    ] == 6
    assert config["visual_protocol"]["minimum_passing_rounds"] == 2
    assert config["operator_refit_allowed"] is False
    assert config["strength_retuning_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_az1_validates_exact_disjoint_evidence_lineage() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2az1_density_residual_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    validated = validate_contract(ROOT, config)
    assert len(validated["eligible_ids"]) == 16
    assert len(
        {validated["source_rows"][key]["make"] for key in validated[
            "eligible_ids"
        ]}
    ) == 9
    assert validated["overlap_count"] == 0


def test_az1_rejects_strength_drift() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2az1_density_residual_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    config["candidate"]["opponent_strength"] = 0.36
    with pytest.raises(ValueError, match="frozen boundary drift"):
        validate_contract(ROOT, config)


def test_az1_visual_builder_is_repeat_deterministic(tmp_path: Path) -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2az1_density_residual_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    candidate_dir = (
        ROOT
        / "outputs/u5_r2az1_density_residual_fresh_confirmation_v1/run_a"
    )
    first = build_fresh_visual_evidence(
        root=ROOT,
        config=config,
        candidate_dir=candidate_dir,
        output_dir=tmp_path / "first",
    )
    second = build_fresh_visual_evidence(
        root=ROOT,
        config=config,
        candidate_dir=candidate_dir,
        output_dir=tmp_path / "second",
    )
    assert first["private_mapping_sha256"] == second[
        "private_mapping_sha256"
    ]
    assert [row["sha256"] for row in first["blind_sheets"]] == [
        row["sha256"] for row in second["blind_sheets"]
    ]
