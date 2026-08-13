from pathlib import Path

from src.eval.c2lut_complex_led_identifiability_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bx2_c2lut_complex_led_identifiability_d0_v1.json"


def test_complex_led_fixture_is_exact() -> None:
    first = evaluate(load_contract(CONFIG), ROOT)
    assert first == evaluate(load_contract(CONFIG), ROOT)
    assert first["metrics"]["row_count"] == 24
