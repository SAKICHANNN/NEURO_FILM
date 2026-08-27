from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_p276_luckyhdr_private_runtime_d0 import (
    P276Error,
    _git_blob_sha1,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p276_luckyhdr_private_runtime_d0_v1.json"
EVIDENCE = ROOT / "docs/evidence/P276_LUCKYHDR_PRIVATE_RUNTIME_D0_RESULT.json"


def test_p276_contract_is_bounded_and_rights_limited() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_BINARY_ACQUISITION_OR_PIXEL_DECODE"
    assert len(config["objects"]) == config["limits"]["object_count"] == 11
    assert sum(item["bytes"] for item in config["objects"].values()) == config["limits"]["total_bytes"]
    assert config["rights"]["product_dependency"] is False
    assert config["runtime"]["exposures"] == [1.0, 3.7, 30.0]


def test_p276_git_blob_identity_helper() -> None:
    assert _git_blob_sha1(b"test\n") == "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"


def test_p276_invalid_order_rejects_before_source_access() -> None:
    with pytest.raises(P276Error, match="order"):
        execute(CONFIG, ROOT, "sideways")


def test_p276_evidence_preserves_runtime_and_claim_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_LUCKYHDR_OFFICIAL_BRACKET_RUNTIME_D0"
    assert evidence["result"]["two_fresh_processes_scientific_exact"] is True
    assert evidence["result"]["output_shape"] == [2048, 1536, 3]
    assert evidence["result"]["target_reads"] == 0
    assert evidence["result"]["metric_runs"] == 0
    assert evidence["rights_and_product"]["hdr_ground_truth_quality"] is False
    assert evidence["rights_and_product"]["candidate_3"] is False
