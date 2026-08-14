from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.eval.scanner_setting_transfer import evaluate, load_contract, write_report

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6ar_chin_scanner_setting_transfer_v1.json"


def test_public_factorial_replays_and_rejects_universal_transfer(tmp_path: Path) -> None:
    report = evaluate(load_contract(CONTRACT))
    assert report["aggregate_gains"] == {"multi_exposure": 0.27, "infrared": 0.18}
    assert report["metrics"] == {
        "minimum_multi_exposure_gain": 0.02,
        "maximum_multi_exposure_gain": 0.58,
        "minimum_infrared_gain": -0.1,
        "maximum_infrared_gain": 0.62,
        "maximum_cross_scanner_absolute_contrast_error": 0.505,
    }
    assert report["gate_results"]["published_aggregate_replay"] is True
    assert report["gate_results"]["multi_exposure_direction"] is True
    assert report["gate_results"]["infrared_direction"] is False
    assert report["gate_results"]["cross_scanner_transfer"] is False
    assert report["automatic_pass"] is False
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert write_report(report, first) == write_report(evaluate(load_contract(CONTRACT)), second)
    assert first.read_bytes() == second.read_bytes()


def test_complete_factorial_is_fail_closed(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    contract["observations"].pop()
    path = tmp_path / "bad.json"
    import json

    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported P6AR contract"):
        load_contract(path)


def test_transfer_candidate_can_pass_when_setting_effects_are_shared() -> None:
    contract = copy.deepcopy(load_contract(CONTRACT))
    for row in contract["observations"]:
        row["dynamic_range"] = (
            3.0
            + (0.2 if row["scanner"] == "slide" else 0.0)
            + (0.1 if row["software"] == "2" else 0.0)
            + (0.27 if row["multi_exposure"] else 0.0)
            + (0.18 if row["infrared"] else 0.0)
        )
    report = evaluate(contract)
    assert report["automatic_pass"] is True
    assert all(report["gate_results"].values())
