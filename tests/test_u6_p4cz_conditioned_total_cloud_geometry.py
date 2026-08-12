from pathlib import Path

import pytest

from src.eval.conditioned_total_cloud_geometry import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cz_conditioned_total_cloud_geometry_v1.json"


def test_p4cz_contract() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p4cz-conditioned-total-cloud-geometry-v1"
    )


def test_p4cz_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_p4cz_parent_drift(tmp_path: Path) -> None:
    payload = CONTRACT.read_text().replace(
        '"required_decision": "close_cross_layer_cloud_geometry_without_rescue"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "c.json"
    path.write_text(payload)
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
