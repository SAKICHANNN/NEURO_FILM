from __future__ import annotations

from pathlib import Path

from src.eval.exposure_grain_coupling import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9k_exposure_grain_coupling_v1.json"


def test_formal_exposure_grain_coupling_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["nonzero_offset_different_fraction"] == 1.0


def test_matched_control_has_exact_neutral_identity() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["measurements"]["zero_offset_candidate_control_error"] == 0.0
