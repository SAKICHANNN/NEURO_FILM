from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.historical_blind_preference_feasibility import (
    HistoricalPreferenceFeasibilityError,
    audit_files,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bn8_historical_blind_preference_feasibility_v1.json"


def test_live_historical_blind_evidence_passes_feasibility() -> None:
    report = audit_files(root=ROOT, config_path=CONFIG)
    assert report["pass"] is True
    assert report["measurements"]["experiment_count"] == 8
    assert report["measurements"]["unique_source_count"] == 62


def test_repeated_rounds_are_collapsed_to_effective_units() -> None:
    report = audit_files(root=ROOT, config_path=CONFIG)
    measurements = report["measurements"]
    assert measurements["raw_blind_votes"] > measurements[
        "effective_source_experiment_units"
    ]
    assert len(report["effective_units"]) == measurements[
        "effective_source_experiment_units"
    ]


def test_evidence_hash_drift_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["evidence_inputs"][0]["sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(HistoricalPreferenceFeasibilityError, match="hash drift"):
        audit_files(root=ROOT, config_path=path)


def test_single_arm_decoded_evidence_fails_closed(tmp_path: Path) -> None:
    decision = {
        "experiment_id": "synthetic",
        "confirmed_severe_artifact_count": 0,
        "decoded_choices": [
            {"source_id": "s1", "selected_arm": "a"},
            {"source_id": "s2", "selected_arm": "a"},
        ],
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    import hashlib

    decision_sha = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["evidence_inputs"] = [
        {
            "path": str(decision_path.relative_to(ROOT))
            if decision_path.is_relative_to(ROOT)
            else str(decision_path),
            "sha256": decision_sha,
            "parser": "decoded_choices",
        }
    ]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(HistoricalPreferenceFeasibilityError, match="fewer than two"):
        audit_files(root=Path("/"), config_path=config_path)
