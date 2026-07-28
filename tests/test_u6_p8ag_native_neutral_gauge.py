from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_gauge_conformance import (
    run_native_gauge_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8ag_native_neutral_gauge_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ag_contract_binds_parent_and_native_sources() -> None:
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
    assert config["domain_order"][1] == "neutral-axis gauge"
    assert config["double_counting_control"][
        "physical_parameters_reused_as_ao6_parameters"
    ] is False


def test_p8ag_native_neutral_gauge_conformance(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_gauge_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert report["independent_build_dll_sha_exact"]
    assert report["knot_counts"] == [1025, 1025, 1025]
    assert report["oracle"][
        "maximum_absolute_error_vs_python_float32"
    ] <= report["oracle"]["tolerance"]
    assert report["oracle"]["in_place_byte_exact"]
    assert report["oracle"]["invalid_input_output_unchanged"]
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AH")
