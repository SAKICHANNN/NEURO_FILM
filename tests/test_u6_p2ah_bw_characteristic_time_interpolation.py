from pathlib import Path

from src.eval.bw_characteristic_time_interpolation import (
    load_contract,
    run_audit,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ah_bw_characteristic_time_interpolation_v1.json"


def test_p2ah_leave_one_time_out_replays_exact(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["held_row_count"] == 2
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    assert write_report(first, a) == write_report(second, b)
    assert a.read_bytes() == b.read_bytes()
