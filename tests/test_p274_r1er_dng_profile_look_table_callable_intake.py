from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p274_r1er_dng_profile_look_table_callable_intake import (
    P274Error,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p274_r1er_dng_profile_look_table_callable_intake_v1.json"
EVIDENCE = ROOT / "docs/evidence/P274_R1ER_DNG_PROFILE_LOOK_TABLE_CALLABLE_INTAKE_RESULT.json"
PRODUCER = ROOT.parent / "追色"


def test_p274_contract_binds_isolated_package() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIXTURE_DESERIALIZATION_OR_CALLABLE_IMPORT"
    assert config["callable"]["id"] == "zhuise.dng-profile-look-table-callable.v1"
    assert config["callable"]["stage_order"][-2:] == [
        "ProfileLookTable",
        "ProfileToneCurve",
    ]
    assert config["producer_summary_bindings"]["real_fuji_dimensions"] == [36, 8, 16]


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
def test_p274_source_locked_fixture_executes() -> None:
    report = execute(CONFIG, PRODUCER, "forward")
    assert report["status"] == "PASS_PRIVATE_R1ER_DNG_PROFILE_LOOK_TABLE_CALLABLE_INTAKE"
    assert all(report["scientific"]["gates"].values())


def test_p274_rejects_invalid_order() -> None:
    with pytest.raises(P274Error, match="order"):
        execute(CONFIG, PRODUCER, "sideways")


def test_p274_evidence_binds_formal_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_R1ER_DNG_PROFILE_LOOK_TABLE_CALLABLE_INTAKE"
    assert evidence["rights_and_product"]["candidate_3"] is False
