from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_ao6_display_conformance import (
    run_native_ao6_display_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8al_native_ao6_display_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8al_contract_binds_parent_and_native_sources() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["profile_compiler_config"]) == config[
        "profile_compiler_config_sha256"
    ]
    for category in ("sources", "headers"):
        for relative, expected in config["native_sources"][category].items():
            assert _sha256(ROOT / relative) == expected
    boundary = config["stage_boundary"]
    assert boundary["hard_clip_in_residual"] is False
    assert boundary["operator_refit"] is False


def test_p8al_native_ao6_display_conformance(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_ao6_display_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert report["independent_display_build_dll_sha_exact"]
    assert report["partition_output_byte_exact"]
    assert report["invalid_input_final_output_unchanged"]
    oracle = report["oracle"]
    assert oracle["maximum_output_error"] <= config["gates"][
        "maximum_output_error"
    ]
    assert 0.0 <= oracle["minimum_output"] <= oracle["maximum_output"] <= 1.0
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AM")
