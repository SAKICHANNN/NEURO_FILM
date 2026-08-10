from __future__ import annotations

import json
from pathlib import Path

from src.eval.temporal_exposure_development_order import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9f_exposure_development_order_v1.json"


def test_formal_exposure_development_order_audit_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["maximum_correct_order_implied_stop_error"] < 1e-10
    assert report["measurements"]["wrong_order_p95_implied_stop_span"] > 0.12


def test_contract_forbids_post_development_flicker() -> None:
    contract = load_contract(CONTRACT)
    forbidden = " ".join(contract["forbidden"])
    assert "after development" in forbidden
    assert "density offset" in contract["negative_control"]
    assert json.dumps(contract, allow_nan=False)
