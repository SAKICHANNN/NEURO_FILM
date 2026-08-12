from pathlib import Path

import pytest

from src.eval.cross_layer_cloud_geometry import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cy_cross_layer_cloud_geometry_v1.json"


def test_p4cy_contract() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p4cy-cross-layer-cloud-geometry-v1"
    )


def test_p4cy_frozen_evaluation() -> None:
    r = evaluate(ROOT, CONTRACT)
    assert r["automatic_pass"] is all(r["stable"]["gates"].values())


def test_p4cy_parent_drift(tmp_path: Path) -> None:
    s = CONTRACT.read_text().replace(
        '"required_decision": "retain_shared_component_poisson_as_explicit_cross_layer_reference_primitive"',
        '"required_decision": "wrong"',
    )
    p = tmp_path / "c.json"
    p.write_text(s)
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, p)
