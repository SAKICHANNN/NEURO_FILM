from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_end_to_end_conformance import (
    run_native_end_to_end_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8am_native_end_to_end_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8am_contract_binds_parent_and_no_double_count() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["profile_compiler_config"]) == config[
        "profile_compiler_config_sha256"
    ]
    receipts = config["no_double_count_receipts"]
    assert receipts["physical_sensitometry_used_once"]
    assert receipts["ao6_density_base_used_once_downstream"]
    assert receipts["ao6_t15_c35_residual_used_once_downstream"]
    assert receipts["physical_parameters_reused_as_ao6_parameters"] is False
    assert receipts["ao6_parameters_claimed_as_physical_truth"] is False


def test_p8am_native_end_to_end_conformance(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_end_to_end_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert report["all_stage_partition_rows_byte_exact"]
    assert report["maximum_output_error_vs_python_reference"] <= config[
        "gates"
    ]["maximum_output_error"]
    assert report["no_double_count_receipts"] == config[
        "no_double_count_receipts"
    ]
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AN")
