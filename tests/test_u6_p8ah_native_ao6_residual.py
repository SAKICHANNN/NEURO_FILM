from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_ao6_residual_conformance import (
    run_native_ao6_residual_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8ah_native_ao6_residual_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ah_contract_binds_parent_and_native_sources() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["profile_compiler_config"]) == config[
        "profile_compiler_config_sha256"
    ]
    native = config["native_source"]
    assert _sha256(ROOT / native["source"]) == native["source_sha256"]
    assert _sha256(ROOT / native["header"]) == native["header_sha256"]
    assert config["stage_boundary"]["computes_source_context"] is False
    assert config["stage_boundary"]["refits_operator_per_image"] is False
    assert config["stage_boundary"]["hard_clipping_allowed"] is False


def test_p8ah_native_ao6_residual_conformance(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_ao6_residual_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert report["independent_build_dll_sha_exact"]
    assert report["tone_strength"] == 0.15
    assert report["chroma_strength"] == 0.35
    oracle = report["oracle"]
    assert oracle["maximum_output_absolute_error"] <= oracle[
        "tolerance"
    ]
    assert oracle["maximum_tone_scale_absolute_error"] <= oracle[
        "tolerance"
    ]
    assert oracle["maximum_chroma_scale_absolute_error"] <= oracle[
        "tolerance"
    ]
    assert oracle["in_place_byte_exact"]
    assert oracle["invalid_input_output_unchanged"]
    assert 0.0 <= oracle["minimum_output"] <= oracle["maximum_output"] <= 1.0
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AI")
