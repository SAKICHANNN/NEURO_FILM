from pathlib import Path

from src.eval.trix_hybrid_thomas_amplitude import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2al_trix_hybrid_thomas_amplitude_v1.json"


def test_p2al_hybrid_replays_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["partition_exact"] is True
