from __future__ import annotations

import json

import pytest

from scripts.audit_p226_r1cv_rec2100_pq_runtime_compatibility import (
    CONFIG,
    TARGET,
    build_p226_parity_fixture,
    validate_config,
)
from src.preprocess.ocio_aces2_output import target_display_view


def test_contract_is_frozen_and_preserves_r1cs_closure() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["preserved_boundaries"]["r1cs_status"].startswith("FAIL_CLOSED")
    assert config["preserved_boundaries"]["analytic_adapter_used"] is False
    assert config["execution"]["durable_media_outputs"] == 0
    assert config["execution_amendment"]["formal_reports_before_amendment"] == 0
    assert config["execution_amendment"]["output_range_gate_unchanged"] is True


def test_contract_binds_distinct_exact_ocio_runtimes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["producer"]["required_opencolorio_version"] == "2.5.0"
    assert config["consumer"]["required_opencolorio_version"] == "2.5.2"
    assert len(config["producer"]["opencolorio_wheel_sha256"]) == 64
    assert len(config["producer"]["config_sha256"]) == 64


def test_private_target_maps_to_exact_official_display_view() -> None:
    assert target_display_view(TARGET) == (
        "Rec.2100-PQ - Display",
        "ACES 2.0 - HDR 1000 nits (P3 D65)",
    )


def test_wrong_contract_schema_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["schema"] = "wrong"
    with pytest.raises(RuntimeError, match="schema differs"):
        validate_config(config)


def test_amended_fixture_preserves_rows_and_is_nonnegative_float32() -> None:
    fixture = build_p226_parity_fixture()
    assert fixture.shape == (986, 3)
    assert fixture.dtype.name == "float32"
    assert fixture.flags.c_contiguous
    assert fixture.min() >= 0.0
