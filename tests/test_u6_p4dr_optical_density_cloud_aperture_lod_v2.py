from pathlib import Path

import pytest

from src.eval.optical_density_cloud_aperture_lod_v2 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4dr_optical_density_cloud_aperture_lod_v2.json"


def test_contract_parents() -> None:
    assert load_contract(ROOT, CONTRACT)["experiment_id"].startswith("u6.p4dr-")


def test_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_parent_drift(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(CONTRACT.read_text().replace("close_optical", "wrong_optical"))
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
