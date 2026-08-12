from pathlib import Path

import numpy as np
import pytest

from src.eval.cross_layer_aperture_lod import evaluate, load_contract
from src.film_physics.cross_layer_aperture_lod import block_aperture_mean

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4db_cross_layer_aperture_lod_v1.json"


def test_p4db_contract() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p4db-cross-layer-aperture-lod-v1"
    )


def test_p4db_block_rejects_shape() -> None:
    with pytest.raises(ValueError):
        block_aperture_mean(np.zeros((3, 4, 3)), 2)


def test_p4db_evaluation() -> None:
    r = evaluate(ROOT, CONTRACT)
    assert r["automatic_pass"] is all(r["stable"]["gates"].values())


def test_p4db_parent_drift(tmp_path: Path) -> None:
    s = CONTRACT.read_text().replace(
        '"required_decision": "retain_cross_layer_gaussian_cloud_lod_reference"',
        '"required_decision": "wrong"',
    )
    p = tmp_path / "c.json"
    p.write_text(s)
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, p)
