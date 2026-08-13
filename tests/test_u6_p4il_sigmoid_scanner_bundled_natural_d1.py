from pathlib import Path

from src.eval.sigmoid_scanner_bundled_natural_d1 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4il_sigmoid_scanner_bundled_natural_d1_v1.json"


def test_sigmoid_scanner_natural_d1_is_exact() -> None:
    first = evaluate(load_contract(CONFIG), ROOT)
    assert first == evaluate(load_contract(CONFIG), ROOT)
    assert first["metrics"]["row_count"] == 5
