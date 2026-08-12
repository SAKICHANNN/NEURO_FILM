from pathlib import Path

from src.eval.native_cloud_chain_performance_v2 import evaluate

ROOT=Path(__file__).resolve().parents[1]


def test_p4ej_native_cloud_performance(tmp_path: Path) -> None:
    assert evaluate(ROOT,ROOT/"configs/u6_p4ej_native_cloud_chain_performance_v2.json",tmp_path)["automatic_pass"] is True
