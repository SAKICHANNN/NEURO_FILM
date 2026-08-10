from __future__ import annotations

from pathlib import Path

from src.eval.scanner_glare_spatial_context import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6zb_scanner_glare_spatial_context_v1.json"


def test_formal_scanner_glare_spatial_context_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    lifts = report["measurements"]["candidate_center_lifts"]
    assert lifts["near"] > lifts["intermediate"] > lifts["far"] > 0.0


def test_equal_energy_controls_are_context_blind() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    measurements = report["measurements"]
    assert measurements["global_control_context_span"] == 0.0
    assert measurements["pointwise_control_context_span"] == 0.0
