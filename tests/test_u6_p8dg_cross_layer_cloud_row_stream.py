from pathlib import Path

import pytest

from src.eval.cross_layer_cloud_row_stream import evaluate, load_contract
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_cross_layer_cloud_profile_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8dg_cross_layer_cloud_row_stream_v1.json"


def test_contract_parent() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p8dg-cross-layer-cloud-row-stream-v1"
    )


def test_runtime_rejects_product_profile() -> None:
    from src.eval.cross_layer_cloud_row_stream import _profile

    profile = _profile(ROOT)
    product = CrossLayerCloudReferenceProfile(
        profile.count_profile,
        profile.gaussian_sigma_pixels_cmy,
        profile.gaussian_truncate,
        profile.aperture_factors,
        profile.parent_evidence_sha256,
        True,
    )
    with pytest.raises(ValueError):
        next(
            iter_cross_layer_cloud_profile_rows(
                product, (32, 32), seed=1, row_tile_height=8
            )
        )


def test_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_parent_drift(tmp_path: Path) -> None:
    text = CONTRACT.read_text().replace(
        "retain_versioned_cross_layer_cloud_reference_profile", "wrong"
    )
    path = tmp_path / "contract.json"
    path.write_text(text)
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
