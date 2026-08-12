from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p8de_streamed_native_thomas_scale import _validate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8de_streamed_native_thomas_scale_v1.json"


def test_p8de_contract_is_frozen_before_formal_execution() -> None:
    contract, p8dd = _validate(CONTRACT)
    assert contract["schema"] == "neuro_film.u6_p8de_streamed_native_thomas_scale_contract.v1"
    assert contract["fixture"]["tile_rows"] == 512
    assert contract["gates"]["minimum_rss_reduction_bytes"] == 134217728
    assert contract["gates"]["maximum_rss_ratio_to_p8dd"] == 0.86
    assert contract["gates"]["maximum_wall_ratio_to_p8dd"] == 1.15
    assert contract["gates"]["maximum_decoded_code_error"] == 1
    assert contract["gates"]["maximum_changed_decoded_fraction"] == 0.001
    assert p8dd["fixture"]["height"] == 3000


def test_p8de_contract_is_strict_json() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["decision_if_fail"] == "retain_p8dd_full_fft_spatial_native_thomas_runtime"
