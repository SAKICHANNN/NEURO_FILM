from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.color_match.shared_hdr_dpct_payload import (
    SharedHdrDpctPayloadError,
    apply_shared_hdr_dpct_payload,
    load_shared_hdr_dpct_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p230_r1cz_shared_hdr_payload_consumer_v1.json"
EVIDENCE = ROOT / "docs/evidence/P230_R1CZ_SHARED_HDR_PAYLOAD_CONSUMER_RESULT.json"
PRODUCER = Path("C:/Users/hhvrf/Documents/追色")


def _inputs() -> tuple[bytes, dict]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload = (PRODUCER / config["producer"]["payload_path"]).read_bytes()
    return payload, config["envelope"]


def test_exact_r1cz_payload_loads_and_applies_owned_output() -> None:
    payload, envelope = _inputs()
    bundle = load_shared_hdr_dpct_payload(payload, envelope)
    source = np.asarray([[[0.0, 18.0, 203.0]]], dtype=np.float32)
    before = source.copy()
    output = apply_shared_hdr_dpct_payload(source, bundle)
    assert np.array_equal(source, before)
    assert output.dtype == np.float32
    assert output.flags.c_contiguous and output.flags.owndata
    assert np.isfinite(output).all()
    assert np.all((output >= 0.0) & (output <= 10000.0))


def test_fixture_identity_is_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    levels = np.asarray(config["fixture"]["levels_float32"], dtype=np.float32)
    fixture = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    ).reshape(-1, 1, 3)
    assert list(fixture.shape) == config["fixture"]["shape"]
    assert hashlib.sha256(fixture.tobytes()).hexdigest() == config["fixture"]["sha256"]


@pytest.mark.parametrize("mutation", ["payload", "hash", "format", "bundle"])
def test_integrity_mutations_fail_closed(mutation: str) -> None:
    payload, envelope = _inputs()
    envelope = dict(envelope)
    if mutation == "payload":
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    elif mutation == "hash":
        envelope["payload_sha256"] = "sha256:" + "0" * 64
    elif mutation == "format":
        envelope["payload_format"] = "unsupported"
    else:
        envelope["bundle_id"] = "sha256:" + "0" * 64
    with pytest.raises(SharedHdrDpctPayloadError):
        load_shared_hdr_dpct_payload(payload, envelope)


def test_source_contract_fails_closed() -> None:
    payload, envelope = _inputs()
    bundle = load_shared_hdr_dpct_payload(payload, envelope)
    with pytest.raises(SharedHdrDpctPayloadError, match="float32 HxWx3"):
        apply_shared_hdr_dpct_payload(np.zeros((2, 3), dtype=np.float32), bundle)
    with pytest.raises(SharedHdrDpctPayloadError, match="finite"):
        apply_shared_hdr_dpct_payload(
            np.asarray([[[np.nan, 0.0, 0.0]]], dtype=np.float32), bundle
        )


def test_formal_evidence_binds_pass_without_consumer_mapping() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_R1CZ_SHARED_HDR_PAYLOAD_CONSUMER"
    assert evidence["result"]["maximum_abs_error"] == 0.0
    assert evidence["result"]["rmse"] == 0.0
    assert all(evidence["gates"].values())
    assert evidence["consumer_mapping"] is False
