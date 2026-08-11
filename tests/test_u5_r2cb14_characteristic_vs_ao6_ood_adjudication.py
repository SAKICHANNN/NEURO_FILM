from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.characteristic_vs_ao6_ood_adjudication import (
    CharacteristicVsAo6OodAdjudicationError,
    adjudicate_files,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experiments/u5_r2cb14_characteristic_vs_ao6_ood_v1"


def _run() -> dict[str, object]:
    return adjudicate_files(
        config_path=ROOT / "configs/u5_r2cb14_characteristic_vs_ao6_ood_v1.json",
        observations_path=ROOT
        / "configs/u5_r2cb14_characteristic_vs_ao6_ood_observations_v1.json",
        mapping_receipt_path=ROOT
        / "configs/u5_r2cb14_characteristic_vs_ao6_ood_mapping_receipt_v1.json",
        report_paths=[BASE / "report_b.json", BASE / "report_c.json"],
        render_dir=BASE / "run_b",
        adjudicator_software_commit="0" * 40,
    )


def test_cb14_adjudication_retains_ao6() -> None:
    result = _run()
    assert result["pass"] is False
    assert result["status"] == "candidate_fails_ao6_retain_incumbent"
    assert result["measurements"]["candidate_round_wins"] == 1
    assert result["measurements"]["candidate_aggregate_choices"] == 16
    assert result["measurements"]["candidate_source_majorities"] == 5
    assert result["full_resolution_review_required"] is False


def test_cb14_adjudication_rejects_observation_mutation(tmp_path: Path) -> None:
    source = ROOT / "configs/u5_r2cb14_characteristic_vs_ao6_ood_observations_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["rounds"][0]["choices"][0]["choice"] = "B"
    mutated = tmp_path / "observations.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CharacteristicVsAo6OodAdjudicationError):
        adjudicate_files(
            config_path=ROOT / "configs/u5_r2cb14_characteristic_vs_ao6_ood_v1.json",
            observations_path=mutated,
            mapping_receipt_path=ROOT
            / "configs/u5_r2cb14_characteristic_vs_ao6_ood_mapping_receipt_v1.json",
            report_paths=[BASE / "report_b.json", BASE / "report_c.json"],
            render_dir=BASE / "run_b",
            adjudicator_software_commit="0" * 40,
        )
