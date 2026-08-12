from pathlib import Path

import numpy as np
import pytest

from src.eval.cross_layer_gaussian_cloud_lod import evaluate, load_contract
from src.film_physics.cross_layer_gaussian_cloud import (
    render_cross_layer_gaussian_cloud_density,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4da_cross_layer_gaussian_cloud_lod_v1.json"


def test_p4da_contract() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p4da-cross-layer-gaussian-cloud-lod-v1"
    )


def test_p4da_renderer_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        render_cross_layer_gaussian_cloud_density(
            np.zeros((2, 2, 3), dtype=np.float32),
            mark_optical_density_cmy=(0.1, 0.1, 0.1),
            sigma_pixels_cmy=(1.0, 1.0, 1.0),
        )


def test_p4da_frozen_evaluation() -> None:
    r = evaluate(ROOT, CONTRACT)
    assert r["automatic_pass"] is all(r["stable"]["gates"].values())


def test_p4da_parent_drift(tmp_path: Path) -> None:
    s = CONTRACT.read_text().replace(
        '"required_decision": "retain_shared_component_poisson_as_explicit_cross_layer_reference_primitive"',
        '"required_decision": "wrong"',
    )
    p = tmp_path / "c.json"
    p.write_text(s)
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, p)
