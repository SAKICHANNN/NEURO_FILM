from pathlib import Path

from src.eval.trix_granularity_compatibility import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2aj_trix_granularity_compatibility_v1.json"


def test_p2aj_fixed_candidate_replays() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["measurements"]["repeat_byte_exact"] is True
    assert first["gate_results"]["maximum_median_relative_error"] is False
