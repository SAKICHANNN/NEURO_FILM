from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.nonsquare_nps_window_confirmation import (
    NonsquareNPSWindowConfirmationError,
    evaluate_nonsquare_nps_window_confirmation,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bo_nonsquare_nps_window_confirmation_v1.json"


def test_p4bo_contract_is_fresh_and_frozen() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bo_nonsquare_nps_window_confirmation_contract.v1"
    )
    assert payload["confirmation"]["fresh_seeds"] == [
        2608022401,
        2608022402,
        2608022403,
    ]
    assert payload["confirmation"]["window_shapes"] == [[160, 224], [224, 160]]
    assert payload["confirmation"]["candidate_window"] == "rectangular"
    assert payload["evaluation"]["minimum_each_shape_relative_median_improvement"] == 0.2


def test_p4bo_contract_rejects_shape_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["confirmation"]["window_shapes"] = [[256, 256]]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NonsquareNPSWindowConfirmationError, match="contract drift"):
        load_contract(path)


def test_nonsquare_nps_window_confirmation_formal_decision() -> None:
    report = evaluate_nonsquare_nps_window_confirmation(
        load_contract(CONTRACT), ROOT
    )
    assert report["profile_count"] == 5
    assert report["decision"] in {
        "promote_rectangular_synthetic_nps_measurement_bridge",
        "retain_hann_synthetic_nps_measurement_bridge",
    }
    assert report["gate_results"]["repeat_exact"] is True

