from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8bk_native_standard_content_stress_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8bk_native_standard_content_stress_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8bk_binds_exact_mixed_rights_content_stress_contract() -> None:
    config = json.loads(CONTRACT.read_text())
    decision = json.loads(DECISION.read_text())
    assert decision["contract_sha256"] == _sha256(CONTRACT)
    assert len(config["rows"]) == 4
    portrait = config["rows"][0]
    assert portrait["sha256"] == (
        "85c076f8e44bea7b4bd48193d758bdcc7a6343d7f161e6d21d6c47c62b5de836"
    )
    assert "CC-BY-NC-SA" in portrait["license"]
    assert "evaluation_only" in portrait["allowed_use"]
    assert all(
        "training" not in row["allowed_use"] for row in config["rows"]
    )


def test_p8bk_passes_severe_gate_without_promoting_appeal() -> None:
    decision = json.loads(DECISION.read_text())
    result = decision["result"]
    assert result["status"] == "pass_limited"
    assert result["automatic_gate_pass"]
    assert result["output_code_boundary_fraction"] == [0.0] * 4
    visual = result["autonomous_visual_review"]
    assert visual["confirmed_severe_artifact_count"] == 0
    assert visual["face_structure_and_fine_detail_preserved"]
    assert "orange skin highlights" in visual["appeal_risk"]
    assert not decision["production_default_changed"]
    assert "global convex strengths" in decision["next_leaf"]
