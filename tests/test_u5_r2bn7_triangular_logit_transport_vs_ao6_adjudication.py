from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.fivek_triangular_logit_transport_visual_adjudication import (
    ADAPTIVE_ARM,
    DIRECT_ARM,
    adjudicate_choices,
)
from src.eval.fivek_triangular_logit_transport_vs_ao6_adjudication import (
    adjudicate_files,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_A = ROOT / "outputs/u5_r2bn7_triangular_logit_transport_vs_ao6_v1/run_a"
RUN_B = ROOT / "outputs/u5_r2bn7_triangular_logit_transport_vs_ao6_v1/run_b"


def _mapping(sources: set[str]) -> dict[str, dict[str, str]]:
    return {
        source: {"A": ADAPTIVE_ARM, "B": DIRECT_ARM} for source in sources
    }


def test_direct_choice_gates_require_nineteen_choices() -> None:
    sources = {f"s{index}" for index in range(10)}
    choices = [
        {
            source: ("A" if index < count else "B")
            for index, source in enumerate(sorted(sources))
        }
        for count in (6, 7, 6)
    ]
    result = adjudicate_choices(
        choices_by_round=choices,
        mappings_by_round=[_mapping(sources)] * 3,
        thresholds={
            "maximum_confirmed_severe_artifact_count": 0,
            "minimum_adaptive_round_wins": 2,
            "minimum_adaptive_aggregate_choices": 19,
            "minimum_adaptive_source_majorities": 6,
        },
        confirmed_severe_count=0,
        automatic_pass=True,
        repeat_exact=True,
        baseline_arm=DIRECT_ARM,
    )
    assert result["pass"]
    assert result["aggregate_counts"][ADAPTIVE_ARM] == 19


@pytest.mark.skipif(
    not RUN_A.is_dir() or not RUN_B.is_dir(), reason="ignored BN7 evidence unavailable"
)
def test_frozen_bn7_adjudication_rejects_challenger() -> None:
    result = adjudicate_files(
        root=ROOT,
        config_path=ROOT / "configs/u5_r2bn7_triangular_logit_transport_vs_ao6_v1.json",
        observations_path=ROOT
        / "configs/u5_r2bn7_triangular_logit_transport_vs_ao6_blind_observations_v1.json",
        full_resolution_review_path=ROOT
        / "configs/u5_r2bn7_triangular_logit_transport_vs_ao6_full_resolution_review_v1.json",
        mapping_receipt_path=ROOT
        / "configs/u5_r2bn7_triangular_logit_transport_vs_ao6_mapping_receipt_v1.json",
        render_report_paths=[RUN_A / "report.json", RUN_B / "report.json"],
        mapping_paths=[
            RUN_A / "blind" / f"blind_round_{index}_mapping.json"
            for index in (1, 2, 3)
        ],
        adjudicator_software_commit="a" * 40,
    )
    assert not result["pass"]
    assert result["adaptive_round_wins"] == 0
    assert result["aggregate_counts"][ADAPTIVE_ARM] == 9
    assert result["adaptive_source_majorities"] == 3
