from __future__ import annotations

from pathlib import Path

from src.eval.portra400_bounded_operator_d1 import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cham7_contract_is_bounded_and_disclosed() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cham7_portra400_bounded_operator_d1_v1.json"
    )
    assert (
        contract["models"]["candidate"]["family"] == "fixed_neutral_trilinear_log_odds"
    )
    assert contract["models"]["candidate"]["fitted_scalar_count"] == 24
    assert contract["disclosure"]["cham6_result_seen_before_freeze"] is True
    assert contract["disclosure"]["independent_confirmation_claim_allowed"] is False
    assert "not independent confirmation" in contract["claim_ceiling"]
