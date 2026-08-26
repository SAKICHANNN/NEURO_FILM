from __future__ import annotations

import json

import pytest

from scripts.audit_p228_r1cy_paired_hdr_handoff_artifacts import (
    CONFIG,
    validate_config,
)

EVIDENCE = (
    CONFIG.parents[1]
    / "docs"
    / "evidence"
    / "P228_R1CY_PAIRED_HDR_HANDOFF_ARTIFACT_AUDIT_RESULT.json"
)


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


def test_formal_evidence_preserves_missing_payload_failure() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_R1CY_HANDOFF_PAYLOAD_NOT_PERSISTED"
    assert evidence["consumer"]["reports_byte_exact"] is True
    assert evidence["consumer"]["report_bytes_each"] == 4148
    assert evidence["consumer"]["stable_identity"] == (
        "sha256:8049c7d8e5171b2afee1d56a1d4c47e8aa7c8a15c05960584e69b5bb887ae1fc"
    )
    assert evidence["artifact_facts"]["tracked_exact_payload_files"] == []
    assert evidence["artifact_facts"]["eval_exact_payload_files"] == []
    assert evidence["gates"]["payload_bytes_persisted_and_sha_exact"] is False
    assert evidence["gates"]["consumer_can_construct_bundle_without_rerunning_build"] is False
    assert evidence["consumer_mapping"] is False
