from __future__ import annotations

import json
from pathlib import Path

from src.eval.native_standard_portable import (
    COMPONENT_NAMES,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bo_portable_native_standard_v1.json"


def test_p8bo_contract_binds_full_strength_sources() -> None:
    contract = json.loads(CONFIG.read_text())
    validate_contract(ROOT, contract)
    assert tuple(contract["components"]) == COMPONENT_NAMES
    assert contract["oracle"]["strength"] == 1.0
    assert contract["production_default_changed"] is False
    assert tuple(contract["components"]["display"]["link_sources"]) == (
        "native/film_physics/nf_ao6_base_f32_v3.c",
        "native/film_physics/nf_ao6_residual_f32_v2.c",
        "native/film_physics/nf_ao6_display_f32_v4.c",
    )


def test_p8bo_platform_claims_remain_non_runtime() -> None:
    contract = json.loads(CONFIG.read_text())
    assert set(contract["targets"]["android"]) == {
        "arm64-v8a",
        "x86_64",
    }
    assert set(contract["targets"]["apple_object_only"]) == {
        "macos-arm64",
        "ios-arm64",
    }
    assert "no simulator/device/runtime/product" in contract[
        "claim_ceiling"
    ]
