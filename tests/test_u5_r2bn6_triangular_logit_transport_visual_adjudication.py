from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_triangular_logit_transport_visual_adjudication import (
    ADAPTIVE_ARM,
    GLOBAL_ARM,
    FiveKTriangularVisualAdjudicationError,
    adjudicate_choices,
    adjudicate_files,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_A = ROOT / "outputs/u5_r2bn6_triangular_logit_transport_visual_product_value_v1/run_a"
RUN_B = ROOT / "outputs/u5_r2bn6_triangular_logit_transport_visual_product_value_v1/run_b"


def _mapping(sources: set[str]) -> dict[str, dict[str, str]]:
    return {
        source: {"A": ADAPTIVE_ARM, "B": GLOBAL_ARM} for source in sources
    }


def test_choice_gates_pass_exact_thresholds() -> None:
    sources = {f"s{index}" for index in range(12)}
    choices = [
        {
            source: ("A" if index < count else "B")
            for index, source in enumerate(sorted(sources))
        }
        for count in (6, 9, 8)
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
    assert result["adaptive_round_wins"] == 2
    assert result["aggregate_counts"][ADAPTIVE_ARM] == 23


@pytest.mark.skipif(
    not RUN_A.is_dir() or not RUN_B.is_dir(), reason="ignored BN6 evidence unavailable"
)
def test_frozen_bn6_adjudication_passes() -> None:
    result = adjudicate_files(
        root=ROOT,
        config_path=ROOT
        / "configs/u5_r2bn6_triangular_logit_transport_visual_product_value_v1.json",
        observations_path=ROOT
        / "configs/u5_r2bn6_triangular_logit_transport_blind_observations_v1.json",
        full_resolution_review_path=ROOT
        / "configs/u5_r2bn6_triangular_logit_transport_full_resolution_review_v1.json",
        mapping_receipt_path=ROOT
        / "configs/u5_r2bn6_triangular_logit_transport_mapping_receipt_v1.json",
        render_report_paths=[RUN_A / "report.json", RUN_B / "report.json"],
        mapping_paths=[
            RUN_A / "blind" / f"blind_round_{index}_mapping.json"
            for index in (1, 2, 3)
        ],
        adjudicator_software_commit="a" * 40,
    )
    assert result["pass"]
    assert result["adaptive_round_wins"] == 2
    assert result["aggregate_counts"][ADAPTIVE_ARM] == 23
    assert result["adaptive_source_majorities"] == 8


@pytest.mark.skipif(
    not RUN_A.is_dir() or not RUN_B.is_dir(), reason="ignored BN6 evidence unavailable"
)
def test_frozen_bn6_rejects_observation_drift(tmp_path: Path) -> None:
    path = ROOT / "configs/u5_r2bn6_triangular_logit_transport_blind_observations_v1.json"
    observations = json.loads(path.read_text(encoding="utf-8"))
    observations["rounds"][0]["choices"][0]["choice"] = "B"
    drifted = tmp_path / "observations.json"
    drifted.write_text(json.dumps(observations), encoding="utf-8")
    with pytest.raises(FiveKTriangularVisualAdjudicationError):
        adjudicate_files(
            root=ROOT,
            config_path=ROOT
            / "configs/u5_r2bn6_triangular_logit_transport_visual_product_value_v1.json",
            observations_path=drifted,
            full_resolution_review_path=ROOT
            / "configs/u5_r2bn6_triangular_logit_transport_full_resolution_review_v1.json",
            mapping_receipt_path=ROOT
            / "configs/u5_r2bn6_triangular_logit_transport_mapping_receipt_v1.json",
            render_report_paths=[RUN_A / "report.json", RUN_B / "report.json"],
            mapping_paths=[
                RUN_A / "blind" / f"blind_round_{index}_mapping.json"
                for index in (1, 2, 3)
            ],
            adjudicator_software_commit="a" * 40,
        )
