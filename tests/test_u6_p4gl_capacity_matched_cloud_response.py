import json
from pathlib import Path

import numpy as np

from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import optical_density_capacity_cmy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gl_capacity_matched_cloud_response_v1.json"


def test_p4gl_freezes_capacity_profile_before_execution():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["parents"]["capacity_profile"]["required_profile_identity"] == (
        "9d9d86e1f2fc695d431810cbe06cbe9749be601838d4fc4cdfed18b63f13879a"
    )
    assert contract["candidate"]["additional_rate_multiplier"] == 1.0
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["execution"]["p4gh_chart_pixels_read"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert contract["gates"]["maximum_absolute_mean_transmittance_bias"] == 0.02


def test_p4gl_exact_profile_capacity_covers_sensitometry_density():
    report = evaluate_capacity(
        ROOT, ROOT / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    profile = CrossLayerCloudReferenceProfile.from_payload(report["compiled_profile"])
    capacity = np.asarray(optical_density_capacity_cmy(profile))
    maximum = np.asarray(report["stable"]["maximum_sensitometry_density_cmy"])
    assert profile.identity() == json.loads(CONTRACT.read_text())["parents"][
        "capacity_profile"
    ]["required_profile_identity"]
    assert np.all(capacity >= maximum)
