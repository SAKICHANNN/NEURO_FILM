from pathlib import Path

from src.eval.historical_wiener_boolean_density_compatibility import (
    load_contract,
    run_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p2ap_historical_wiener_boolean_density_compatibility_v1.json"
)


def test_p2ap_boolean_density_compatibility_replays() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    assert first["measurements"]["parameter_refit_count_zero"] is True
    assert first["measurements"]["rgb_image_transform_count_zero"] is True
