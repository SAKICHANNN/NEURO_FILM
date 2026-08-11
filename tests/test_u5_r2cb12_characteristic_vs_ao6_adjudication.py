from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.characteristic_vs_ao6_adjudication import (
    CharacteristicVsAo6AdjudicationError,
    adjudicate_files,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experiments/u5_r2cb12_characteristic_vs_ao6_fresh_v1"


def _run() -> dict[str, object]:
    return adjudicate_files(
        root=ROOT,
        config_path=ROOT / "configs/u5_r2cb12_characteristic_vs_ao6_fresh_v1.json",
        observations_path=ROOT
        / "configs/u5_r2cb12_characteristic_vs_ao6_fresh_observations_v1.json",
        review_path=ROOT
        / "configs/u5_r2cb12_characteristic_vs_ao6_full_resolution_review_v1.json",
        mapping_receipt_path=ROOT
        / "configs/u5_r2cb12_characteristic_vs_ao6_mapping_receipt_v1.json",
        report_paths=[BASE / "report_run4.json", BASE / "report_run5.json"],
        adjudicator_software_commit="0" * 40,
    )


def test_cb12_adjudication_passes_frozen_gates() -> None:
    result = _run()
    assert result["pass"] is True
    assert result["status"] == "candidate_beats_ao6_open_gold_stress"
    assert result["measurements"]["candidate_round_wins"] == 3
    assert result["measurements"]["candidate_aggregate_choices"] == 34
    assert result["measurements"]["candidate_source_majorities"] == 12
    assert all(result["gates"].values())


def test_cb12_adjudication_rejects_observation_mutation(tmp_path: Path) -> None:
    source = ROOT / "configs/u5_r2cb12_characteristic_vs_ao6_fresh_observations_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["rounds"][0]["choices"][0]["choice"] = "B"
    mutated = tmp_path / "observations.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CharacteristicVsAo6AdjudicationError):
        adjudicate_files(
            root=ROOT,
            config_path=ROOT / "configs/u5_r2cb12_characteristic_vs_ao6_fresh_v1.json",
            observations_path=mutated,
            review_path=ROOT
            / "configs/u5_r2cb12_characteristic_vs_ao6_full_resolution_review_v1.json",
            mapping_receipt_path=ROOT
            / "configs/u5_r2cb12_characteristic_vs_ao6_mapping_receipt_v1.json",
            report_paths=[BASE / "report_run4.json", BASE / "report_run5.json"],
            adjudicator_software_commit="0" * 40,
        )
