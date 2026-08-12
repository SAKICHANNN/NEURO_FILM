from pathlib import Path

from src.eval.native_cloud_typed_chain_parity import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_p4ee_typed_chain_parity(tmp_path: Path) -> None:
    report = evaluate(
        ROOT,
        ROOT / "configs/u6_p4ee_native_cloud_typed_chain_parity_v1.json",
        tmp_path,
    )
    assert report["automatic_pass"] is True
    assert all(report["stable"]["gates"].values())
