from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_shared_characteristic_shape import (
    SharedCharacteristicShapeError,
    evaluate_shared_shape,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2u_shared_characteristic_shape_v1.json"
TRACE = ROOT / "configs/data/kodak_vision3_characteristic_curve_pixels_v1.json"


def test_contract_rejects_tail_gate_relaxation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_p95_rmse_improvement_fraction"] = -1.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SharedCharacteristicShapeError, match="contract drift"):
        load_contract(path)


def test_contract_rejects_foreign_trace_path(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["parent"]["trace_path"] = "../trace.json"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SharedCharacteristicShapeError, match="repository-relative"):
        load_contract(path)


@pytest.mark.skipif(not TRACE.is_file(), reason="exact Kodak characteristic trace unavailable")
def test_shared_shape_is_repeatable_and_closes_blue_tail(tmp_path: Path) -> None:
    del tmp_path
    contract = load_contract(CONTRACT)
    first_report, first_bundle = evaluate_shared_shape(contract, ROOT)
    second_report, second_bundle = evaluate_shared_shape(contract, ROOT)
    assert first_report == second_report
    assert first_bundle == second_bundle
    assert not first_report["shared_template_pass"]
    assert first_report["candidate_win_count"] == 8
    assert first_report["wins_by_channel"] == {"blue": 2, "green": 3, "red": 3}
    assert first_report["gate_results"]["every_channel_win_count"]
    assert not first_report["gate_results"]["p95_rmse_improvement"]
    assert all(
        value
        for key, value in first_report["gate_results"].items()
        if key != "p95_rmse_improvement"
    )
    assert first_report["decision"] == (
        "close_empirical_shared_template_and_retain_only_canonical_gauge_plus_raw_source_observations"
    )
