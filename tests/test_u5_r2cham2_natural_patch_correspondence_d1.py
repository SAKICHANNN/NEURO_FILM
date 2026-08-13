from pathlib import Path

from src.eval.natural_patch_correspondence_d1 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cham2_natural_patch_correspondence_d1_v1.json"


def test_natural_patch_correspondence_is_exact() -> None:
    first = evaluate(load_contract(CONFIG), ROOT)
    assert first == evaluate(load_contract(CONFIG), ROOT)
    assert first["metrics"]["row_count"] == 5
