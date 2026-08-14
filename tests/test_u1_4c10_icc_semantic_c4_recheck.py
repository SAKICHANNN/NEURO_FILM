from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.icc_semantic_c4_recheck import (
    ICCSemanticC4RecheckError,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c10_icc_semantic_c4_recheck_v1.json"


def test_contract_reuses_c4_without_gate_or_parameter_change() -> None:
    contract = load_contract(CONTRACT)
    assert contract["experiment_id"] == "U1.4C10"
    assert contract["execution"]["mapping_render_and_gates"] == "exact-u1-4c4-evaluator"
    assert contract["execution"]["parameter_change_allowed"] is False
    assert contract["execution"]["gate_change_allowed"] is False
    assert "not independent data" in contract["claim_ceiling"]


def test_contract_rejects_hash_drift(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_bytes(CONTRACT.read_bytes() + b" ")
    with pytest.raises(ICCSemanticC4RecheckError, match="contract hash drift"):
        load_contract(changed)
