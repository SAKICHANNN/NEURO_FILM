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
