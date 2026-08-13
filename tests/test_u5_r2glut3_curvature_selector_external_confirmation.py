from pathlib import Path

from src.eval.curvature_selector_external_confirmation import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2glut3_curvature_selector_external_confirmation_v1.json"


def test_external_confirmation_is_exact_and_bounded() -> None:
    first = evaluate(load_contract(CONFIG), ROOT)
    assert first == evaluate(load_contract(CONFIG), ROOT)
    assert first["metrics"]["row_count"] == 8
    assert all(row["gaussian_rmse"] > 0 and row["lattice_rmse"] > 0 for row in first["rows"])
