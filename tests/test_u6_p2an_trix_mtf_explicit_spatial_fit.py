from pathlib import Path

from src.eval.trix_mtf_explicit_spatial_fit import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2an_trix_mtf_explicit_spatial_fit_v1.json"


def test_p2an_held_frequency_decision_replays() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["gate_results"]["maximum_development_absolute_error"] is False
    assert first["measurements"]["rgb_image_transform_count_zero"] is True
    assert (
        first["parent_profile_identity"]
        == "c88425fc2642aa64c3bb73d5fa88022e537e3ed18611409e93b1e27ccd555d73"
    )
