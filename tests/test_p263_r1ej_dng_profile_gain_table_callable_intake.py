from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p263_r1ej_dng_profile_gain_table_callable_intake import (
    P263Error,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p263_r1ej_dng_profile_gain_table_callable_intake_v1.json"
PRODUCER = ROOT.parent / "追色"


def test_p263_contract_binds_complete_isolated_package() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIXTURE_DESERIALIZATION_OR_CALLABLE_IMPORT"
    assert set(config["artifacts"]) == {
        "contract",
        "schema",
        "arithmetic_core",
        "callable",
        "fixture",
        "execution_lock",
        "evidence",
    }
    assert config["callable"]["id"] == "zhuise.dng-profile-gain-table-callable.v1"


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
def test_p263_source_locked_fixture_executes() -> None:
    report = execute(CONFIG, PRODUCER, "forward")
    assert report["status"] == "PASS_PRIVATE_R1EJ_DNG_PROFILE_GAIN_TABLE_CALLABLE_INTAKE"
    assert all(report["scientific"]["gates"].values())


def test_p263_rejects_invalid_order() -> None:
    with pytest.raises(P263Error, match="order"):
        execute(CONFIG, PRODUCER, "sideways")
