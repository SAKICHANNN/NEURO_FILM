from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p320_r1gs_hasselblad_x2d_no_copy_intake import execute

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p320_r1gs_hasselblad_x2d_no_copy_intake_v1.json"
PRODUCER = ROOT.parent / "追色"


def test_p320_contract_is_frozen_and_claim_is_narrow() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIRST_CONSUMER_DECODE"
    assert config["protocol"] == "zhuise.hasselblad-x2d-dual-container-unpack.v1"
    assert len(config["sources"]) == 4
    assert "no generic 3FR/FFF/Hasselblad RAW" in config["claim_ceiling"]
    assert "candidate-3 change" in config["claim_ceiling"]


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_p320_exact_no_copy_intake(order: str) -> None:
    report = execute(CONFIG, PRODUCER, order)
    assert report["status"] == "PASS_PRIVATE_R1GS_HASSELBLAD_X2D_NO_COPY_INTAKE"
    assert all(report["scientific"]["gates"].values())
    assert report["rights_and_product"]["consumer_core_copied"] is False
    assert report["rights_and_product"]["producer_runner_executed"] is False
    assert report["rights_and_product"]["producer_reference_decoder_executed"] is False
    assert report["rights_and_product"]["product_mapping"] is False
