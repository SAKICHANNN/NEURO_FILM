from pathlib import Path

from src.eval.structural_region_matching_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cham1_structural_region_matching_d0_v1.json"


def test_structural_region_matching_is_exact() -> None:
    first = evaluate(load_contract(CONFIG), ROOT)
    assert first == evaluate(load_contract(CONFIG), ROOT)
    assert first["metrics"]["row_count"] == 24
