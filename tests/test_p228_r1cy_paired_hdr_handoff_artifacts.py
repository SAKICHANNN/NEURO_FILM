from __future__ import annotations

import json

import pytest

from scripts.audit_p228_r1cy_paired_hdr_handoff_artifacts import CONFIG, validate_config


def test_contract_is_frozen_before_consumer_mapping() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["status"] == "FROZEN_AFTER_R1CY_HANDOFF_BEFORE_CONSUMER_MAPPING"
    assert config["bounded_artifact_scope"]["payload_hash_only_is_not_payload"] is True
    assert config["bounded_artifact_scope"]["unbounded_output_or_data_scan"] is False


def test_contract_requires_exact_payload_bytes_not_reconstruction() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["producer"]["payload_bytes"] == 4376
    assert config["producer"]["payload_sha256"].startswith("sha256:")
    assert "without reconstructing" in config["decision"]["fail"]


def test_wrong_schema_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["schema"] = "wrong"
    with pytest.raises(RuntimeError, match="schema differs"):
        validate_config(config)
