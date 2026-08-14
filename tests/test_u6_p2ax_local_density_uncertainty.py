from pathlib import Path

from src.eval.local_density_uncertainty import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ax_local_density_uncertainty_v1.json"


def test_p2ax_uncertainty_audit_replays_and_decides_from_frozen_gates() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is all(first["gate_results"].values())
    assert first["measurements"]["minimum_developed_density"] > 0.0
    assert len(first["aggregate_rows"]) == 16
    assert len(first["realizations"]) == 12
