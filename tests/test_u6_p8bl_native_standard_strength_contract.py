from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8bl_native_standard_strength_sweep_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bl_native_standard_strength_sweep_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8bl_decision_binds_frozen_strength_sweep() -> None:
    contract = json.loads(CONTRACT.read_text())
    decision = json.loads(DECISION.read_text())
    assert decision["contract_sha256"] == _sha256(CONTRACT)
    assert contract["strengths"] == [0.65, 0.8, 1.0]
    assert contract["strength_domain"] == "encoded-display-srgb"
    result = decision["result"]
    assert result["automatic_gate_pass"]
    assert result["all_twelve_outputs_zero_boundary"]
    assert result["all_twelve_outputs_zero_new_boundary"]
    assert result["full_strength_reproduced_exactly"]


def test_p8bl_selects_strongest_reduced_global_challenger_only() -> None:
    decision = json.loads(DECISION.read_text())
    visual = decision["result"]["autonomous_visual_review"]
    assert visual["preferred_global_development_strength"] == 0.8
    assert visual["confirmed_severe_artifact_count"] == 0
    assert not visual["population_preference_claimed"]
    assert "do not add content-aware routing" in decision["decision"]
    assert not decision["production_default_changed"]
    assert decision["next_leaf"].startswith("U6.P8BM")
