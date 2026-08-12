from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p8dd_native_thomas_fft_spatial_scale import _validate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8dd_native_thomas_fft_spatial_scale_v1.json"


def test_p8dd_contract_is_frozen_before_formal_execution() -> None:
    contract, _p1, _p3d, _package = _validate(CONTRACT)
    assert contract["schema"] == "neuro_film.u6_p8dd_native_thomas_fft_spatial_scale_contract.v1"
    assert contract["gates"]["maximum_relative_exposure_error"] == 1e-6
    assert contract["gates"]["maximum_p99_exposure_ulp_error"] == 1.0
    assert contract["gates"]["maximum_exposure_ulp_error"] == 8.0
    assert contract["gates"]["maximum_decoded_code_error"] == 1
    assert contract["gates"]["maximum_changed_decoded_fraction"] == 0.001
    assert contract["gates"]["minimum_wall_speedup_over_direct"] == 1.5


def test_p8dd_contract_is_strict_json() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["decision_if_fail"] == "retain_p8dc_direct_fullframe_spatial_runtime"
