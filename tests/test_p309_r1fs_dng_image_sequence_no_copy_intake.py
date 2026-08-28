from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p309_r1fs_dng_image_sequence_no_copy_intake import (
    P309Error,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p309_r1fs_dng_image_sequence_no_copy_intake_v1.json"
PRODUCER = ROOT.parent / "追色"


def test_p309_contract_is_source_locked_and_no_copy() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_PRODUCER_OBJECT_IMPORT_OR_SOURCE_METADATA_READ"
    assert config["protocol"] == "zhuise.dng-image-sequence-info-group.v1"
    assert len(config["sources"]) == 3
    assert [row["index"] for row in config["sources"]] == [1, 2, 3]
    assert "src/preprocess" not in json.dumps(config)


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
def test_p309_source_locked_sequence_executes() -> None:
    report = execute(CONFIG, PRODUCER, "forward")
    assert report["status"] == "PASS_PRIVATE_R1FS_DNG_IMAGE_SEQUENCE_NO_COPY_INTAKE"
    assert all(report["scientific"]["gates"].values())
    assert report["rights_and_product"]["consumer_core_copied"] is False
    assert report["rights_and_product"]["image_pixels_decoded"] == 0


def test_p309_rejects_invalid_order() -> None:
    with pytest.raises(P309Error, match="order"):
        execute(CONFIG, PRODUCER, "sideways")
