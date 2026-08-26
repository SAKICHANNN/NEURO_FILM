from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p224_r1cr_aces2_xyz_runtime_compatibility import (
    CONFIG,
    validate_config,
)


def test_contract_is_frozen_after_r1cs_and_excludes_analytic_pq_rescue() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["excluded_closed_route"]["producer_leaf"] == "R1CS"
    assert config["excluded_closed_route"]["status"] == "FAIL_CLOSED"
    assert "does not encode XYZ" in config["excluded_closed_route"]["exclusion"]
    assert config["execution"]["media_reads"] == 0


def test_contract_binds_distinct_producer_and_consumer_runtime_versions() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["producer"]["required_opencolorio_version"] == "2.5.0"
    assert len(config["producer"]["opencolorio_wheel_sha256"]) == 64
    assert config["consumer"]["required_opencolorio_version"] == "2.5.2"
    assert config["producer"]["output_builtin"].endswith("1000nit-P3-D65_2.0")


def test_wrong_schema_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["schema"] = "wrong"
    with pytest.raises(RuntimeError, match="schema differs"):
        validate_config(config)


def test_tracked_evidence_preserves_r1cs_closure_and_exact_xyz_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root
            / "docs/evidence/P224_R1CR_ACES2_XYZ_RUNTIME_COMPATIBILITY_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "PASS_PRIVATE_R1CR_XYZ_CROSS_RUNTIME_COMPATIBILITY"
    assert evidence["consumer"]["reports_byte_exact"] is True
    assert evidence["mechanism_facts"]["maximum_absolute_difference_nits"] == 0.0
    assert evidence["mechanism_facts"]["rmse_difference_nits"] == 0.0
    assert evidence["closed_route_preservation"]["producer_leaf"] == "R1CS"
    assert evidence["closed_route_preservation"]["status"] == "FAIL_CLOSED"
    assert evidence["execution_amendment"]["failed_attempt_report_count"] == 0
    assert evidence["decision"]["does_not_open"]
