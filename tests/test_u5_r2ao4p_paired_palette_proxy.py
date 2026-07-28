from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.run_u5_r2ao4p_balica_paired_palette_proxy import (
    CONFIG_SHA256,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao4p_balica_paired_palette_proxy_v1.json"


def test_ao4p_config_identity_and_pair_semantics() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    assert [group["expected_pair_count"] for group in config["groups"]] == [
        47,
        56,
    ]
    assert config["sample"]["paired_channel_order"] == [
        "film_rgb",
        "reference_rgb",
    ]
    assert config["operator_fitting_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_ao4p_config_hash_fail_closed() -> None:
    try:
        load_config(CONFIG, expected_sha256="f" * 64)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("AO4P accepted a foreign config identity")
