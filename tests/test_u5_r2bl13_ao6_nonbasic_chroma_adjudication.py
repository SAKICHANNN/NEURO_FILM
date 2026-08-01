from __future__ import annotations

import json
from pathlib import Path

from src.eval.ao6_nonbasic_chroma_adjudication import (
    adjudicate_ao6_nonbasic_chroma,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_bl13_evidence_closes_unstable_preference() -> None:
    config = json.loads(
        (
            ROOT / "configs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_v1.json"
        ).read_text(encoding="utf-8")
    )
    base = ROOT / "outputs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_v1"
    result = adjudicate_ao6_nonbasic_chroma(
        config=config,
        observations_path=(
            ROOT
            / "configs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_observations_v1.json"
        ),
        run_a=base / "formal_a",
        run_b=base / "formal_b",
        blind_dir=base / "blind_v1",
    )
    assert result["automatic_gate_pass"] is True
    assert result["confirmed_severe_artifact_count"] == 0
    assert result["candidate_choices_by_round"] == [13, 8, 9]
    assert result["passing_rounds"] == 1
    assert result["visual_gate_pass"] is False
    assert result["decision"] == "close_bl13_preference_retain_ao6"
    assert result["product_integration_opened"] is False
