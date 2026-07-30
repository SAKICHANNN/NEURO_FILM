import json
from pathlib import Path

import pytest

from src.eval.filmmatch_paired_source import sha256_file
from src.eval.filmmatch_baseline_blind import ARMS
from src.eval.filmmatch_baseline_blind import adjudicate_filmmatch_blind


def test_ax17_freezes_three_fixed_arms_and_strict_blind_gate() -> None:
    config = json.loads(
        Path(
            "configs/u5_r2ax17_filmmatch_fixed_baseline_blind_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert tuple(config["arms"]) == ARMS
    assert config["blind_gate"]["rounds"] == 3
    assert config["blind_gate"]["minimum_overall_round_wins"] == 2
    assert config["blind_gate"]["minimum_round_wins_vs_ao6"] == 2
    assert config["gates"]["additional_rounds_after_reveal_allowed"] is False
    assert config["gates"]["threshold_relaxation_allowed"] is False


def test_ax17_decision_closes_failed_global_challenger() -> None:
    decision = json.loads(
        Path(
            "configs/"
            "u5_r2ax17_filmmatch_fixed_baseline_blind_decision_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["blind_gate_passed"] is False
    assert decision["automatic_summary"]["filmmatch_choice_count"] == 3
    assert decision["automatic_summary"]["b0_choice_count"] == 15
    assert decision["automatic_summary"]["confirmed_severe_artifact_count"] == 0
    assert decision["promotion_opened"] is False
    assert decision["decision"].startswith("close_filmmatch_cap070")


def test_ax17_adjudication_rejects_mapping_identity_drift(
    tmp_path: Path,
) -> None:
    mappings = []
    for round_index in (1, 2, 3):
        path = tmp_path / f"blind_round_{round_index}_mapping.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "source_id": "source",
                        "A": "fixed_b0",
                        "B": "fixed_ao6_t15_c35",
                        "C": "filmmatch_cap070",
                    }
                ]
            ),
            encoding="utf-8",
        )
        mappings.append(
            {
                "round": round_index,
                "mapping_sha256": sha256_file(path),
            }
        )
    (tmp_path / "build_receipt.json").write_text(
        json.dumps(
            {
                "experiment_id": "experiment",
                "artifacts": mappings,
            }
        ),
        encoding="utf-8",
    )
    observations = tmp_path / "observations.json"
    observations.write_text(
        json.dumps(
            {
                "mapping_unread_when_recorded": True,
                "rounds": [
                    {"round": index, "votes": {"source": "A"}}
                    for index in (1, 2, 3)
                ],
                "confirmed_severe_artifact_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "blind_round_2_mapping.json").write_text(
        "[]", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mapping identity drift"):
        adjudicate_filmmatch_blind(
            root=tmp_path,
            config={
                "experiment_id": "experiment",
                "blind_gate": {
                    "minimum_overall_round_wins": 2,
                    "minimum_total_choice_share": 0.4,
                    "minimum_round_wins_vs_ao6": 2,
                    "minimum_choice_share_vs_ao6": 0.55,
                },
                "claim_ceiling": "test",
            },
            build_dir=tmp_path,
            observations_path=observations,
        )
