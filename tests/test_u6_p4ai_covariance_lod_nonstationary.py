from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_covariance_lod_nonstationary import CovarianceLodNonstationaryError, evaluate_covariance_lod_nonstationary, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ai_covariance_lod_nonstationary_v1.json"


def test_contract_mutation_fails_closed(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["gates"]["maximum_cell_variance_ratio"] = 2.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CovarianceLodNonstationaryError, match="contract drift"):
        load_contract(path)


def test_parent_mutation_fails_closed(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["parents"]["p4ah_decision_sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CovarianceLodNonstationaryError, match="parent mismatch"):
        evaluate_covariance_lod_nonstationary(load_contract(path), ROOT)


def test_frozen_evaluation_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_covariance_lod_nonstationary(contract, ROOT)
    second = evaluate_covariance_lod_nonstationary(contract, ROOT)
    assert first == second
    assert first["passed"] is all(first["checks"].values())
