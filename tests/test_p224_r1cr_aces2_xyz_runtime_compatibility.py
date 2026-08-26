from __future__ import annotations

import json

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
