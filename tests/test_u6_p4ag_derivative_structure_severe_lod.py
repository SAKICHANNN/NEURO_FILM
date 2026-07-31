from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_derivative_structure_severe_lod import (
    DerivativeStructureSevereLodError,
    evaluate_severe_lod,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ag_derivative_structure_severe_lod_v1.json"


def test_contract_rejects_lod_gate_mutation(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["gates"]["maximum_lod_variance_ratio"] = 2.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DerivativeStructureSevereLodError, match="contract drift"):
        load_contract(path)


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["parents"]["p4af_decision_sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DerivativeStructureSevereLodError, match="parent mismatch"):
        evaluate_severe_lod(load_contract(path), ROOT)


def test_frozen_evaluation_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_severe_lod(contract, ROOT)
    second = evaluate_severe_lod(contract, ROOT)
    assert first == second
    assert first["decision_key"] in {"severe_fail", "severe_pass_lod_fail", "all_pass"}
