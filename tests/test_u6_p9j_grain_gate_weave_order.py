from __future__ import annotations

from pathlib import Path

from src.eval.grain_gate_weave_order import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9j_grain_gate_weave_order_v1.json"


def test_formal_grain_gate_weave_order_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["nonzero_offset_different_frame_fraction"] == 1.0


def test_contract_keeps_emulsion_grain_before_scanner_motion() -> None:
    contract = load_contract(CONTRACT)
    assert contract["correct_order"] == [
        "developed_emulsion_structure",
        "transmittance",
        "scanner_gate_weave_sampling",
    ]
