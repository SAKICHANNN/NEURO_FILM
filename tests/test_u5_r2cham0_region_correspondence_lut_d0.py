from pathlib import Path

from src.eval.region_correspondence_lut_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cham0_region_correspondence_lut_d0_v1.json"


def test_region_correspondence_d0_is_exact() -> None:
    first = evaluate(load_contract(CONFIG))
    assert first == evaluate(load_contract(CONFIG))
    assert first["metrics"]["row_count"] == 24
