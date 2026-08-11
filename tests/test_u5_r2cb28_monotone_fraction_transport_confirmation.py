from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.monotone_fraction_transport_confirmation import (
    MonotoneFractionTransportConfirmationError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cb28_contract_binds_exact_cb27_operator() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb28_monotone_fraction_transport_confirmation_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB28"
    assert contract["operator"]["implementation_sha256"] == (
        "2ef3908055e7eea6bc4e6d04a21d8b2f0e6875fa309f6329326e8bf646b6a7ea"
    )
    assert contract["population"]["source_count_exact"] == 12


def test_cb28_rejects_parent_decision_drift(tmp_path: Path) -> None:
    contract_path = (
        ROOT / "configs/u5_r2cb28_monotone_fraction_transport_confirmation_v1.json"
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["parents"]["cb27_required_status"] = "forged"
    with pytest.raises(
        MonotoneFractionTransportConfirmationError, match="decision drift"
    ):
        evaluate(payload, ROOT, tmp_path / "out")
