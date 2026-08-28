from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p312_r1fv_dng_profile_gain_table2_stage_no_copy_intake import execute

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p312_r1fv_dng_profile_gain_table2_stage_no_copy_intake_v1.json"
PRODUCER = ROOT.parent / "追色"


def test_p312_contract_is_frozen_and_gamma_claim_is_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIRST_CONSUMER_STAGE_EXECUTION"
    assert config["execution"]["gamma2_discriminating"] is False
    assert "no general non-unit-gamma claim" in config["claim_ceiling"]


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_p312_exact_stage_intake(order: str) -> None:
    report = execute(CONFIG, PRODUCER, order)
    assert report["status"] == (
        "PASS_PRIVATE_R1FV_DNG_PROFILE_GAIN_TABLE2_STAGE_NO_COPY_INTAKE"
    )
    assert all(report["scientific"]["gates"].values())
    assert report["rights_and_product"]["consumer_core_copied"] is False
    assert report["rights_and_product"]["general_gamma_claim"] is False
    assert report["rights_and_product"]["product_mapping"] is False
