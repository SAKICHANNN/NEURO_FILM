from pathlib import Path

from src.eval.trix_hybrid_thomas_amplitude import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2al_trix_hybrid_thomas_amplitude_v1.json"
CORRECTION = (
    ROOT / "configs/u6_p2al1_trix_hybrid_thomas_identity_correction_v1.json"
)


def test_p2al_hybrid_replays_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["partition_exact"] is True
    assert first["stable_evidence_id"] == (
        "1b673fbe63623379730d2bae35e84708ea577cbbd65fddf163eb474d65bb7a8f"
    )


def test_p2al1_corrects_identity_without_changing_measurements() -> None:
    legacy = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    corrected = run_audit(root=ROOT, contract=load_contract(CORRECTION))
    assert corrected["automatic_pass"] is True
    assert corrected["scalar_profile_identity"] == (
        "482714d55fc2591b4c748ceb61118a1db4b29fb1deba0156e97999b62e897c8b"
    )
    assert corrected["measurement_energy"] == legacy["measurement_energy"]
    assert corrected["point_scale"] == legacy["point_scale"]
    assert corrected["measurements"] == legacy["measurements"]
    assert corrected["profile_identity_correction"]["numeric_mechanism_changed"] is False
