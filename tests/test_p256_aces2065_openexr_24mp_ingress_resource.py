from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p256_aces2065_openexr_24mp_ingress_resource_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_config_binds_unchanged_p251_and_p255() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    for prefix in (
        "contract",
        "p251_module",
        "p251_config",
        "p251_evidence",
        "p255_config",
        "p255_evidence",
        "p255_native_source",
        "p255_runner",
    ):
        path = ROOT / bindings[f"{prefix}_path"]
        assert path.stat().st_size == bindings[f"{prefix}_bytes"]
        assert _sha256(path) == bindings[f"{prefix}_sha256"]
    assert config["status"] == "FROZEN_BEFORE_RUNNER_OR_PIXEL_EXECUTION"
    assert config["input"]["openexr_sha256"] == (
        "eeced074ea933cd02421d491bc54c78fdba5704caa9bfe3705d47d410a01f275"
    )


def test_resource_and_numeric_gates_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["gates"]["maximum_reader_process_tree_rss_bytes"] == 2**31
    assert config["gates"]["maximum_reader_wall_seconds"] == 120.0
    assert config["expected_output"]["maximum_ap1_roundtrip_error"] == 2**-20
    assert config["input"]["height"] * config["input"]["width"] == 24_000_000
    assert "Candidate 3 remains closed at 2/3" in config["claim_ceiling"]
