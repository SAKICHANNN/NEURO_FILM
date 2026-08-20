from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.real_film.three_stock_admission_matrix import (
    STOCK_IDS,
    ThreeStockAdmissionError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "sf3_a3_three_stock_admission_matrix_v1.json"


def test_current_matrix_fails_closed_for_every_stock() -> None:
    report = evaluate(load_contract(CONTRACT), ROOT)
    assert [row["film_stock_id"] for row in report["stock_rows"]] == STOCK_IDS
    assert not report["all_three_stocks_k1_fit_admitted"]
    assert all(not row["k1_fit_admitted"] for row in report["stock_rows"])
    assert report["current_baseline"]["experiment_id"] == "RF3.D0"
    assert report["current_baseline"]["ao6_role"].endswith("baseline_only")
    assert report["decision"].startswith("RETAIN_RF3_D0")


def test_one_complete_shared_lane_admits_all_three_stocks() -> None:
    contract = copy.deepcopy(load_contract(CONTRACT))
    controlled = next(
        row
        for row in contract["source_lanes"]
        if row["source_id"] == "project_owned_controlled_capture"
    )
    for key in contract["k1_admission_gates"]:
        controlled[key] = True
    report = evaluate(contract, ROOT)
    assert report["all_three_stocks_k1_fit_admitted"]
    assert all(row["k1_fit_admitted"] for row in report["stock_rows"])
    assert report["decision"] == contract["decision_if_all_stocks_admitted"]


def test_contract_rejects_stock_scope_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["required_stock_ids"].pop()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ThreeStockAdmissionError, match="contract drift"):
        load_contract(path)


def test_bound_evidence_hash_drift_fails(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    project = tmp_path / "project"
    binding = contract["evidence_bindings"][0]
    evidence = project / binding["path"]
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(ThreeStockAdmissionError, match="evidence drift"):
        evaluate(contract, project)
