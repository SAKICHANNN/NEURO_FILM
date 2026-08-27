from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p272_illum_sensor_mapping_source_eligibility import (
    P272Error,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p272_illum_sensor_mapping_source_eligibility_v1.json"
EVIDENCE = (
    ROOT
    / "docs/evidence/P272_ILLUM_SENSOR_MAPPING_SOURCE_ELIGIBILITY_RESULT.json"
)


def test_p272_contract_is_frozen_before_data_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_DATASET_OBJECT_REQUEST"
    assert config["head_commit"] == "f649f85cddaafdcdfd92bc32cea2415d3b72fd1e"
    assert set(config["artifacts"]) == {"readme", "license"}


def test_p272_source_closes_before_data_request() -> None:
    report = execute(CONFIG, "forward")
    assert report["status"] == "FAIL_CLOSED_SOURCE_RIGHTS_AND_MANIFEST"
    scientific = report["scientific"]
    assert scientific["dataset_object_requests"] == 0
    assert scientific["pixel_reads"] == 0
    assert scientific["gates"]["official-identity"] is True
    assert scientific["gates"]["commercial-product-research-rights"] is False


def test_p272_rejects_invalid_order() -> None:
    with pytest.raises(P272Error, match="order"):
        execute(CONFIG, "sideways")


def test_p272_evidence_binds_closed_zero_data_decision() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_SOURCE_RIGHTS_AND_MANIFEST"
    assert evidence["formal_reports"][0]["sha256"] == evidence["formal_reports"][1][
        "sha256"
    ]
    assert evidence["result"]["dataset_object_requests"] == 0
    assert evidence["rights_and_product"]["candidate_3"] is False
