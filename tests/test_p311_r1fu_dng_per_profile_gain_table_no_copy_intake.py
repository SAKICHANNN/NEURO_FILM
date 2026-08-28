from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p311_r1fu_dng_per_profile_gain_table_no_copy_intake import (
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p311_r1fu_dng_per_profile_gain_table_no_copy_intake_v1.json"
PRODUCER = ROOT.parent / "追色"


def test_p311_contract_is_frozen_and_narrow() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIRST_CONSUMER_SAMPLE_PARSE"
    assert config["experiment_id"] == "P311"
    assert config["protocol"] == "zhuise.dng-per-profile-gain-table-selection.v1"
    assert "candidate-3 change" in config["claim_ceiling"]
    assert len(config["expected"]["profiles"]) == 2


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_p311_executes_exact_no_copy_profile_association(order: str) -> None:
    report = execute(CONFIG, PRODUCER, order)
    assert report["status"] == (
        "PASS_PRIVATE_R1FU_DNG_PER_PROFILE_GAIN_TABLE_NO_COPY_INTAKE"
    )
    assert all(report["scientific"]["gates"].values())
    assert report["rights_and_product"] == {
        "candidate_3": False,
        "consumer_core_copied": False,
        "image_pixels_decoded": 0,
        "product_mapping": False,
        "public_capability": False,
        "sdk_renders_executed": 0,
    }
