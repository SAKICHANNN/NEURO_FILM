from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.physical_native_composed_display_conformance import (
    run_native_composed_display_conformance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8ai_native_composed_display_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ai_contract_binds_parent_and_boundaries() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["profile_compiler_config"]) == config[
        "profile_compiler_config_sha256"
    ]
    assert config["stage_ownership"]["source_context_base"].startswith(
        "external frozen Python"
    )
    assert config["claim_ledger"]["physical_chain_is_stock_calibrated"] is False
    assert config["claim_ledger"]["ao6_is_physical_truth"] is False
    assert config["claim_ledger"][
        "source_context_is_refit_per_partition"
    ] is False


def test_p8ai_native_composed_display_conformance(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    try:
        report = run_native_composed_display_conformance(
            root=ROOT,
            config=config,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        if "MSVC Build Tools are unavailable" in str(error):
            pytest.skip(str(error))
        raise
    fixture = report["fixture"]
    assert fixture["shape"] == [23, 29, 3]
    assert fixture[
        "maximum_encoded_error_vs_python_residual"
    ] <= report["maximum_encoded_error_tolerance"]
    assert len(fixture["partition_rows"]) == 10
    assert all(
        all(
            row[key]
            for key in (
                "physical_byte_exact",
                "gauge_byte_exact",
                "external_base_byte_exact",
                "residual_byte_exact",
            )
        )
        for row in fixture["partition_rows"]
    )
    assert report["claim_ledger"]["ao6_is_physical_truth"] is False
    assert report["production_default_changed"] is False
    assert report["next_leaf"].startswith("U6.P8AJ")
