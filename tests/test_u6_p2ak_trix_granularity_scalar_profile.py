from pathlib import Path

from src.eval.trix_granularity_scalar_profile import load_contract, run_audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ak_trix_granularity_scalar_profile_v1.json"


def test_p2ak_scalar_profile_replays_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["profile"]["render_allowed"] is False
    assert first["profile"]["spatial_nps_status"] == "unidentified"
