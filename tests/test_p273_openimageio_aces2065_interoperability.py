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
