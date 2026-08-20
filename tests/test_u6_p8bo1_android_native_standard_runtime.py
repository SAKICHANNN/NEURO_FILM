from __future__ import annotations

import json
from pathlib import Path

from src.eval.native_standard_android_runtime import (
    P8BO_SOURCE_COMMIT,
    _git_bytes,
    _validate_parent,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8bo1_android_native_standard_runtime_v1.json"
PROBE = ROOT / "native/film_physics/nf_native_standard_android_probe_v1.c"


def test_parent_and_portable_contracts_validate() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent, portable = _validate_parent(ROOT, contract)
    assert parent["result"]["status"] == contract["parent"]["required_result_status"]
    assert portable["schema"] == (
        "neuro_film.u6_p8bo_portable_native_standard_contract.v1"
    )
    assert _git_bytes(
        ROOT,
        P8BO_SOURCE_COMMIT,
        "native/film_physics/nf_physical_domains_f32_v1.h",
    ).startswith(b"#ifndef NF_PHYSICAL_DOMAINS_F32_V1_H")


def test_probe_exercises_complete_component_chain_and_atomic_failure() -> None:
    source = PROBE.read_text(encoding="utf-8")
    required_calls = (
        "nf_ao6_context_f32_update_v2",
        "nf_gaussian_f32_apply_v1",
        "nf_physical_sensitometry_f32_apply_v1",
        "nf_bounded_adjacency_f32_apply_v1",
        "nf_physical_interpretation_f32_apply_v1",
        "nf_neutral_gauge_f32_apply_v1",
        "nf_ao6_display_f32_apply_v4",
    )
    assert all(call in source for call in required_calls)
    assert "invalid_atomic" in source
