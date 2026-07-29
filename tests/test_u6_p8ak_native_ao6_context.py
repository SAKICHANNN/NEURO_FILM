from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_ao6_context_conformance import (
    run_native_ao6_context_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8ak_native_ao6_context_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ak_contract_binds_parent_and_native_sources() -> None:
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
    boundary = config["stage_boundary"]
    assert boundary["partition_invariant"] is True
    assert boundary["operator_refit"] is False


def test_p8ak_native_ao6_context_conformance(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_ao6_context_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert report["independent_context_build_dll_sha_exact"]
    assert report["partition_context_byte_exact"]
    assert report["invalid_update_state_unchanged"]
    oracle = report["oracle"]
    gates = config["gates"]
    assert oracle["maximum_context_mean_error"] <= gates[
        "maximum_context_mean_error"
    ]
    assert oracle["maximum_context_std_error"] <= gates[
        "maximum_context_std_error"
    ]
    assert oracle["maximum_output_error"] <= gates[
        "maximum_output_error"
    ]
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AL")
