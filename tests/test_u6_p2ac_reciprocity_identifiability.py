from __future__ import annotations

from pathlib import Path

from src.eval.reciprocity_identifiability import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ac_reciprocity_identifiability_v1.json"


def test_anchored_design_recovers_parameters_and_confirmation() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    measurements = report["measurements"]
    assert report["automatic_pass"] is True
    assert measurements["maximum_anchored_parameter_absolute_error"] < 0.02
    assert measurements["anchored_confirmation_log_rmse"] < 0.002


def test_unanchored_design_exposes_structural_ill_conditioning() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    measurements = report["measurements"]
    assert measurements["unanchored_scaled_jacobian_condition"] > 1e6
    assert measurements["maximum_unanchored_exponent_absolute_error"] > 0.01
    assert measurements["global_confirmation_log_rmse"] > 0.05
    assert measurements["wrong_time_confirmation_log_rmse"] > 0.02
