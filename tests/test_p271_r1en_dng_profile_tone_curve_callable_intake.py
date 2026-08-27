from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p271_r1en_dng_profile_tone_curve_callable_intake import (
    P271Error,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p271_r1en_dng_profile_tone_curve_callable_intake_v1.json"
EVIDENCE = ROOT / "docs/evidence/P271_R1EN_DNG_PROFILE_TONE_CURVE_CALLABLE_INTAKE_RESULT.json"
PRODUCER = ROOT.parent / "追色"


def test_p271_contract_binds_isolated_package() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIXTURE_DESERIALIZATION_OR_CALLABLE_IMPORT"
    assert set(config["artifacts"]) == {
        "contract",
        "parent_evidence",
        "arithmetic_core",
        "callable",
        "schema",
        "fixture",
        "execution_lock",
        "evidence",
    }
    assert config["callable"]["id"] == "zhuise.dng-profile-tone-curve-callable.v1"


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
def test_p271_source_locked_fixture_executes() -> None:
    report = execute(CONFIG, PRODUCER, "forward")
    assert report["status"] == "PASS_PRIVATE_R1EN_DNG_PROFILE_TONE_CURVE_CALLABLE_INTAKE"
    assert all(report["scientific"]["gates"].values())


def test_p271_rejects_invalid_order() -> None:
    with pytest.raises(P271Error, match="order"):
        execute(CONFIG, PRODUCER, "sideways")


def test_p271_evidence_binds_formal_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_R1EN_DNG_PROFILE_TONE_CURVE_CALLABLE_INTAKE"
    assert evidence["fixed_identity"]["consumer_formal_report_sha256"] == (
        "b4c0f9dc8eeb198e5416f2dff56467b15eea78aa88d22a448c1f1fd02170e52b"
    )
    assert evidence["rights_and_product"]["candidate_3"] is False
