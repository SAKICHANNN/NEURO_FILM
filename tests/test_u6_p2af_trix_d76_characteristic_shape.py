from pathlib import Path

from src.eval.trix_d76_characteristic_shape import (
    load_contract,
    run_audit,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2af_trix_d76_characteristic_shape_v1.json"


def test_p2af_characteristic_shape_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["measurements"]["curve_count"] == 4
    assert first["measurements"]["rgb_image_transform_count_zero"] is True
    assert all(value > 0.0 for value in first["coordinate_uncertainty"].values())
    assert first["automatic_pass"] is False


def test_p2af_report_writer_is_exact(tmp_path: Path) -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert write_report(report, first) == write_report(report, second)
    assert first.read_bytes() == second.read_bytes()
