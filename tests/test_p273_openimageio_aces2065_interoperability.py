from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p273_openimageio_aces2065_interoperability import (
    P273Error,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p273_openimageio_aces2065_interoperability_v1.json"
EVIDENCE = ROOT / "docs/evidence/P273_OPENIMAGEIO_ACES2065_INTEROPERABILITY_RESULT.json"
PRODUCER = ROOT.parent / "追色"


def test_p273_contract_is_frozen_before_oiio_read() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_OPENIMAGEIO_CONTAINER_READ"
    assert config["openimageio_runtime"]["version"] == "3.1.11.0"
    assert config["expected"]["color_interop_id"] == "lin_ap0_scene"


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
def test_p273_exact_container_is_oiio_interoperable() -> None:
    report = execute(CONFIG, PRODUCER, "forward")
    assert report["status"] == "PASS_PRIVATE_OPENIMAGEIO_ACES2065_INTEROPERABILITY"
    assert all(report["scientific"]["gates"].values())


def test_p273_rejects_invalid_order() -> None:
    with pytest.raises(P273Error, match="order"):
        execute(CONFIG, PRODUCER, "sideways")


def test_p273_evidence_binds_exact_interoperability() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_OPENIMAGEIO_ACES2065_INTEROPERABILITY"
    assert evidence["formal_reports"][0]["sha256"] == evidence["formal_reports"][1][
        "sha256"
    ]
    assert evidence["result"]["pixel_f32le_sha256"] == (
        "2b3417e344aa7c26962a109f55d14237a0926d0e03eb46767477b274c4179cb4"
    )
    assert evidence["rights_and_product"]["product_admission"] is False
