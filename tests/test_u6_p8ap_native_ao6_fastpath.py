from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_ao6_fastpath_conformance import (
    run_native_ao6_fastpath_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8ap_native_ao6_fastpath_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ap_contract_binds_parent_and_native_sources() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["profile_compiler_config"]) == config[
        "profile_compiler_config_sha256"
    ]
    context = config["native_sources"]["context"]
    assert _sha256(
        ROOT / "native/film_physics/nf_ao6_context_f32_v2.c"
    ) == context["source_sha256"]
    assert _sha256(
        ROOT / "native/film_physics/nf_ao6_context_f32_v2.h"
    ) == context["header_sha256"]
    for category in ("sources", "headers"):
        for relative, expected in config["native_sources"]["display"][
            category
        ].items():
            assert _sha256(ROOT / relative) == expected


def test_p8ap_native_ao6_fastpath_conformance(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_ao6_fastpath_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    for key in (
        "independent_context_build_byte_exact",
        "independent_display_build_byte_exact",
        "context_v1_v2_byte_exact",
        "context_partition_byte_exact",
        "base_v1_v2_byte_exact",
        "base_v2_in_place_byte_exact",
        "display_v1_v2_byte_exact",
        "display_partition_byte_exact",
        "invalid_context_state_unchanged",
        "invalid_display_final_output_unchanged",
    ):
        assert report[key] is True
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AQ")
