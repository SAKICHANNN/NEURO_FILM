from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p310_r1ft_dng_profile_gain_table2_no_copy_intake import (
    P310Error,
    execute,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p310_r1ft_dng_profile_gain_table2_no_copy_intake_v1.json"
PRODUCER = ROOT.parent / "追色"


def test_p310_contract_is_source_locked_and_no_copy() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == (
        "FROZEN_AFTER_EXCLUDED_UNCOMMITTED_DRY_RUN_BEFORE_FORMAL"
    )
    assert config["excluded_preformal_attempt"]["controls_or_gates_changed"] is False
    assert config["protocol"] == (
        "zhuise.dng-profile-gain-table-map2-encoding-parity.v1"
    )
    assert [row["data_type"] for row in config["sources"]] == [0, 1, 2]
    assert "src/preprocess" not in json.dumps(config)


@pytest.mark.skipif(not PRODUCER.is_dir(), reason="producer repository is absent")
def test_p310_source_locked_parser_executes() -> None:
    report = execute(CONFIG, PRODUCER, "forward")
    assert report["status"] == (
        "PASS_PRIVATE_R1FT_DNG_PROFILE_GAIN_TABLE2_NO_COPY_INTAKE"
    )
    assert all(report["scientific"]["gates"].values())
    assert report["rights_and_product"]["consumer_core_copied"] is False
    assert report["rights_and_product"]["image_pixels_decoded"] == 0


def test_p310_rejects_invalid_order() -> None:
    with pytest.raises(P310Error, match="order"):
        execute(CONFIG, PRODUCER, "sideways")
