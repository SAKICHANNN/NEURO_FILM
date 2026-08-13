from pathlib import Path

from src.eval.c2lut_chromaticity_conditioned_relighting_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bx3_c2lut_chromaticity_conditioned_relighting_d0_v1.json"


def test_chromaticity_conditioned_relighting_is_exact() -> None:
    first = evaluate(load_contract(CONFIG))
    assert first == evaluate(load_contract(CONFIG))
    assert first["metrics"]["row_count"] == 12
