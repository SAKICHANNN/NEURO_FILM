from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import _parent_payloads
from scripts.evaluate_u6_p8db_native_thomas_spatial_scale import _fixture, _validate
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_thomas_input import RelativeLayerLogExposure

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8db_native_thomas_spatial_scale_v1.json"
FULLFRAME_CONTRACT = (
    ROOT / "configs/u6_p8dc_native_thomas_fullframe_spatial_scale_v1.json"
)


def test_p8db_contract_and_fixture_are_domain_valid() -> None:
    contract, _p1, _p3d, _package = _validate(CONTRACT)
    assert contract["gates"]["maximum_process_tree_rss_bytes"] == 1000000000
    prior = ManufacturerCharacteristicPrior.from_dict(_parent_payloads()[1]["prior"])
    exposure = _fixture(prior, 65, 67)
    log_exposure = RelativeLayerLogExposure.from_layer_exposure(exposure).values_chw
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        assert float(log_exposure[channel].min()) >= lower
        assert float(log_exposure[channel].max()) <= upper
    assert json.loads(CONTRACT.read_text(encoding="utf-8"))["decision_if_fail"] == "retain_p8da_without_spatial_runtime_integration"


def test_p8dc_contract_selects_fullframe_mechanism() -> None:
    contract, _p1, _p3d, _package = _validate(FULLFRAME_CONTRACT)
    assert contract["fixture"]["tile_rows"] is None
    assert contract["gates"]["maximum_wall_seconds"] == 45.0
