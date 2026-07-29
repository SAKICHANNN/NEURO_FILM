from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.profile_u6_p8ao_native_end_to_end_phases import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8ao_native_end_to_end_phases_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ao_contract_binds_failed_resource_parent() -> None:
    config = json.loads(CONFIG.read_text())
    validate_contract(config)
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert config["gates"]["minimum_dominant_phase_share"] == 0.35
    assert config["timing_policy"][
        "thresholds_must_not_be_relaxed_after_observation"
    ]
