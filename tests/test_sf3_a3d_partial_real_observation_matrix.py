from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.real_film.three_stock_partial_observation_matrix import (
    GATE_KEYS,
    PartialObservationMatrixError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a3d_partial_real_observation_matrix_v1.json"


def test_current_partial_observations_locate_three_stock_data_gap() -> None:
    report = evaluate(load_contract(CONTRACT), ROOT)
    assert report["stocks_with_any_same_scene_real_colour_observation"] == 2
    assert report["stocks_k1_fit_admitted"] == 0
    assert not report["all_three_stocks_k1_fit_admitted"]
    assert report["stock_rows"][2]["observation_grade"] == "data_gap"
    assert report["first_data_priority"].startswith("first_rights_cleared_same_scene_ektar")
    assert report["decision"].startswith("RETAIN_PARTIAL_REAL_OBSERVATIONS")


def test_complete_independent_observations_admit_all_stocks() -> None:
    contract = copy.deepcopy(load_contract(CONTRACT))
    for lane in contract["observation_lanes"]:
        for key in GATE_KEYS:
            lane[key] = True
    report = evaluate(contract, ROOT)
    assert report["all_three_stocks_k1_fit_admitted"]
    assert report["stocks_k1_fit_admitted"] == 3
    assert report["decision"] == contract["decision_if_all_admitted"]


def test_contract_rejects_evidence_reference_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["observation_lanes"][0]["evidence_ids"] = ["unknown"]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PartialObservationMatrixError, match="invalid observation lane"):
        load_contract(path)


def test_bound_evidence_decision_drift_fails(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    project = tmp_path / "project"
    binding = contract["evidence_bindings"][0]
    evidence = project / binding["path"]
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({"decision": "changed"}), encoding="utf-8")
    contract = copy.deepcopy(contract)
    contract["evidence_bindings"][0]["sha256"] = __import__("hashlib").sha256(evidence.read_bytes()).hexdigest()
    with pytest.raises(PartialObservationMatrixError, match="decision drift"):
        evaluate(contract, project)
