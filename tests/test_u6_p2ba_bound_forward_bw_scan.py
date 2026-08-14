from pathlib import Path

from src.eval.bound_forward_bw_scan import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ba_bound_forward_bw_scan_v1.json"


def test_p2ba_bound_forward_scan_replays_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["arithmetic_inverse_claimed"] is False
