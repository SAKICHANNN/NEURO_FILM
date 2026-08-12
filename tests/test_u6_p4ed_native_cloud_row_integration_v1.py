from __future__ import annotations

from pathlib import Path

from src.eval.native_cloud_row_integration import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_p4ed_native_row_integration(tmp_path: Path) -> None:
    report = evaluate(
        ROOT,
        ROOT / "configs/u6_p4ed_native_cloud_row_integration_v1.json",
        tmp_path,
    )
    assert report["automatic_pass"] is True
    assert all(report["stable"]["gates"].values())
