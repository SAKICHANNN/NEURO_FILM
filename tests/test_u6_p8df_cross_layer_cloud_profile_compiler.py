from pathlib import Path

import pytest

from src.eval.cross_layer_cloud_profile_compiler import (
    compile_profile,
    evaluate,
    load_contract,
)
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8df_cross_layer_cloud_profile_compiler_v1.json"


def test_p8df_contract() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p8df-cross-layer-cloud-profile-compiler-v1"
    )


def test_p8df_roundtrip() -> None:
    p = compile_profile(ROOT, CONTRACT)
    assert (
        CrossLayerCloudReferenceProfile.from_payload(p.to_payload()).identity()
        == p.identity()
    )


def test_p8df_rejects_product_enable() -> None:
    p = compile_profile(ROOT, CONTRACT).to_payload()
    p["product_enabled"] = True
    with pytest.raises(ValueError):
        CrossLayerCloudReferenceProfile.from_payload(p)


def test_p8df_evaluation() -> None:
    r = evaluate(ROOT, CONTRACT)
    assert r["automatic_pass"] is True and all(r["stable"]["gates"].values())
