from pathlib import Path

import numpy as np
import pytest

from src.eval.optical_density_cloud_transmittance_v2 import evaluate, load_contract
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_optical_density_cross_layer_cloud_rows_v2,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4dp_optical_density_cloud_transmittance_v2.json"


def test_contract_parent() -> None:
    assert load_contract(ROOT, CONTRACT)["experiment_id"].startswith("u6.p4dp-")


def test_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_v2_rejects_out_of_capacity() -> None:
    payload = evaluate_capacity(
        ROOT, ROOT / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )["compiled_profile"]
    profile = CrossLayerCloudReferenceProfile.from_payload(payload)
    target = np.full((65, 67, 3), 100.0)
    with pytest.raises(ValueError, match="outside"):
        tuple(
            iter_optical_density_cross_layer_cloud_rows_v2(
                profile, target, seed=1, row_tile_height=31
            )
        )


def test_parent_drift(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(CONTRACT.read_text().replace("close_or_revise", "wrong"))
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
