from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p316_r1gn_fujifilm_raf_no_copy_intake import execute

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p316_r1gn_fujifilm_raf_no_copy_intake_v1.json"
PRODUCER = ROOT.parent / "追色"


def test_p316_contract_is_frozen_and_correction_is_explicit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIRST_CONSUMER_DECODE"
    assert config["protocol"] == "zhuise.fujifilm-raf-dual-layout-unpack.v1"
    assert len(config["sources"]) == 2
    assert config["identity_correction"][
        "science_source_fixture_gates_and_formal_reports_changed"
    ] is False
    assert "no generic RAF" in config["claim_ceiling"]
    assert "candidate-3 change" in config["claim_ceiling"]


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_p316_exact_no_copy_intake(order: str) -> None:
    report = execute(CONFIG, PRODUCER, order)
    assert report["status"] == "PASS_PRIVATE_R1GN_FUJIFILM_RAF_NO_COPY_INTAKE"
    assert all(report["scientific"]["gates"].values())
    correction = report["scientific"]["identity_correction"]
    assert correction["corrected_commit_resolves"] is True
    assert correction["superseded_commit_rejects"] is True
    assert report["rights_and_product"]["consumer_core_copied"] is False
    assert report["rights_and_product"]["producer_runner_executed"] is False
    assert report["rights_and_product"]["product_mapping"] is False
