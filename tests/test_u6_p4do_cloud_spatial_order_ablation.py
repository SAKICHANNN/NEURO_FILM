from pathlib import Path

import pytest

from src.eval.cloud_spatial_order_ablation import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4do_cloud_spatial_order_ablation_v1.json"


def test_contract_parents() -> None:
    assert load_contract(ROOT, CONTRACT)["experiment_id"].startswith("u6.p4do-")


def test_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_parent_drift(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(CONTRACT.read_text().replace("retain_typed", "wrong_typed"))
    with pytest.raises(RuntimeError, match="parent decision drift"):
        load_contract(ROOT, path)
