from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.profile_u6_p8au_native_display_v3_phases import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u6_p8au_native_display_v3_phases_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8au_contract_binds_failed_v3_resource_decision() -> None:
    config = json.loads(CONFIG.read_text())
    validate_contract(config)
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["resource_contract"]) == config[
        "resource_contract_sha256"
    ]
    assert config["gates"]["minimum_dominant_phase_share"] == 0.35
