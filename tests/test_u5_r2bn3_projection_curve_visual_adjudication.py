from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_projection_curve_visual_adjudication import (
    ADAPTIVE_ARM,
    GLOBAL_ARM,
    FiveKProjectionCurveVisualAdjudicationError,
    adjudicate_choices,
    adjudicate_files,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_A = ROOT / "outputs/u5_r2bn3_projection_curve_visual_product_value_v2/run_a"
RUN_B = ROOT / "outputs/u5_r2bn3_projection_curve_visual_product_value_v2/run_b"


def _mapping(sources: set[str], adaptive_label: str = "A") -> dict:
    global_label = "B" if adaptive_label == "A" else "A"
    return {
        source: {
            adaptive_label: ADAPTIVE_ARM,
            global_label: GLOBAL_ARM,
        }
        for source in sources
    }


def test_choice_gates_include_source_majorities() -> None:
    sources = {f"s{index}" for index in range(12)}
    choices = [
        {source: ("A" if index < count else "B") for index, source in enumerate(sorted(sources))}
        for count in (8, 8, 8)
    ]
    result = adjudicate_choices(
        choices_by_round=choices,
        mappings_by_round=[_mapping(sources)] * 3,
        thresholds={
            "maximum_confirmed_severe_artifact_count": 0,
            "minimum_adaptive_round_wins": 2,
            "minimum_adaptive_aggregate_choices": 22,
            "minimum_adaptive_source_majorities": 7,
        },
        confirmed_severe_count=0,
        automatic_pass=True,
        repeat_exact=True,
    )
    assert result["pass"]
    assert result["adaptive_round_wins"] == 3
    assert result["adaptive_source_majorities"] == 8


def test_choice_gates_reject_severe_artifact() -> None:
    sources = {f"s{index}" for index in range(12)}
    result = adjudicate_choices(
        choices_by_round=[{source: "A" for source in sources}] * 3,
        mappings_by_round=[_mapping(sources)] * 3,
        thresholds={
            "maximum_confirmed_severe_artifact_count": 0,
            "minimum_adaptive_round_wins": 2,
            "minimum_adaptive_aggregate_choices": 22,
            "minimum_adaptive_source_majorities": 7,
        },
        confirmed_severe_count=1,
        automatic_pass=True,
        repeat_exact=True,
    )
    assert not result["pass"]
    assert not result["gates"]["maximum_confirmed_severe_artifact_count"]["pass"]


@pytest.mark.skipif(not RUN_A.is_dir() or not RUN_B.is_dir(), reason="ignored BN3 evidence unavailable")
def test_frozen_bn3_v2_adjudication_closes_visual_promotion() -> None:
    result = adjudicate_files(
        root=ROOT,
        config_path=ROOT / "configs/u5_r2bn3_projection_curve_visual_product_value_v2.json",
        observations_path=ROOT / "configs/u5_r2bn3_projection_curve_blind_observations_v2.json",
        full_resolution_review_path=ROOT / "configs/u5_r2bn3_projection_curve_full_resolution_review_v2.json",
        mapping_receipt_path=ROOT / "configs/u5_r2bn3_projection_curve_mapping_receipt_v2.json",
        render_report_paths=[RUN_A / "report.json", RUN_B / "report.json"],
        mapping_paths=[RUN_A / "blind" / f"blind_round_{index}_mapping.json" for index in (1, 2, 3)],
        adjudicator_software_commit="a" * 40,
    )
    assert not result["pass"]
    assert result["adaptive_round_wins"] == 2
    assert result["aggregate_counts"][ADAPTIVE_ARM] == 20
    assert result["adaptive_source_majorities"] == 7
    assert result["gates"]["minimum_adaptive_round_wins"]["pass"]
    assert not result["gates"]["minimum_adaptive_aggregate_choices"]["pass"]
    assert result["gates"]["minimum_adaptive_source_majorities"]["pass"]


@pytest.mark.skipif(not RUN_A.is_dir() or not RUN_B.is_dir(), reason="ignored BN3 evidence unavailable")
def test_frozen_bn3_v2_adjudication_rejects_observation_drift(tmp_path: Path) -> None:
    observations = json.loads((ROOT / "configs/u5_r2bn3_projection_curve_blind_observations_v2.json").read_text(encoding="utf-8"))
    observations["observations"][0]["choice"] = "A"
    drifted = tmp_path / "observations.json"
    drifted.write_text(json.dumps(observations), encoding="utf-8")
    with pytest.raises(FiveKProjectionCurveVisualAdjudicationError):
        adjudicate_files(
            root=ROOT,
            config_path=ROOT / "configs/u5_r2bn3_projection_curve_visual_product_value_v2.json",
            observations_path=drifted,
            full_resolution_review_path=ROOT / "configs/u5_r2bn3_projection_curve_full_resolution_review_v2.json",
            mapping_receipt_path=ROOT / "configs/u5_r2bn3_projection_curve_mapping_receipt_v2.json",
            render_report_paths=[RUN_A / "report.json", RUN_B / "report.json"],
            mapping_paths=[RUN_A / "blind" / f"blind_round_{index}_mapping.json" for index in (1, 2, 3)],
            adjudicator_software_commit="a" * 40,
        )
