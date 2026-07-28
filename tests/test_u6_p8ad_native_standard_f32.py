from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_f32_conformance import (
    run_native_f32_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8ad_native_standard_f32_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ad_contract_binds_parent_and_sources() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["profile_compiler_config"]) == config[
        "profile_compiler_config_sha256"
    ]
    for component in config["component_sources"].values():
        assert _sha256(ROOT / component["source"]) == component[
            "source_sha256"
        ]
        assert _sha256(ROOT / component["header"]) == component[
            "header_sha256"
        ]
    assert config["execution"]["sample_storage"] == "float32"
    assert config["gates"]["invalid_input_output_mutation"] == "none"


@pytest.mark.skipif(
    not (ROOT / "native/film_physics/nf_physical_domains_f32_v1.c").is_file(),
    reason="native source unavailable",
)
def test_p8ad_native_standard_f32_conformance(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_f32_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert report["independent_build_dll_sha_exact"]
    assert report["maximum_absolute_error_vs_float64"] <= report[
        "tolerance"
    ]
    assert all(report["replay"]["failure_atomic"].values())
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AE")
