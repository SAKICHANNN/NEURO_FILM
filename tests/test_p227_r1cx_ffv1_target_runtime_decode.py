from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p227_r1cx_ffv1_target_runtime_decode import CONFIG, validate_config


def test_contract_is_frozen_before_target_pixel_decode() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["status"] == "FROZEN_AFTER_R1CX_PASS_BEFORE_TARGET_PIXEL_DECODE"
    assert config["execution"]["network_reads"] == 0
    assert config["execution"]["media_writes_to_repository"] == 0


def test_contract_binds_two_exact_media_and_independent_runtime() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["producer"]["media"]) == 2
    assert config["producer"]["media_sha256"] == (
        "58556d207db6bdf78db4d2b2938eb1f724cff770c569a93b4abb28d55457e80e"
    )
    assert config["target_runtime"]["ffmpeg_version"] == "8.1"
    assert config["target_runtime"]["provider"] != "producer PyAV runtime"


def test_wrong_schema_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["schema"] = "wrong"
    with pytest.raises(RuntimeError, match="schema differs"):
        validate_config(config)


def test_tracked_evidence_records_exact_target_decode_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root / "docs/evidence/P227_R1CX_FFV1_TARGET_RUNTIME_DECODE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "PASS_PRIVATE_R1CX_FFV1_TARGET_RUNTIME_DECODE"
    assert evidence["consumer"]["reports_byte_exact"] is True
    assert evidence["decode_facts"]["producer_normalized_sequence_exact"] is True
    assert evidence["gates"]["single_video_stream_metadata_exact"] is True
