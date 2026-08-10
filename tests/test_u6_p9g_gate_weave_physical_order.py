from __future__ import annotations

from pathlib import Path

from src.eval.gate_weave_physical_order import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9g_gate_weave_physical_order_v1.json"


def test_formal_gate_weave_physical_order_audit_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["correct_order_partition_byte_exact"] is True
    assert report["measurements"]["median_frame_rmse"] > 0.02


def test_contract_freezes_physical_scanner_order() -> None:
    contract = load_contract(CONTRACT)
    assert contract["correct_order"].endswith("immutable-source bilinear scanner sampling")
    assert "layer exposure" in contract["negative_control"]
    assert any("chained" in item for item in contract["forbidden"])
