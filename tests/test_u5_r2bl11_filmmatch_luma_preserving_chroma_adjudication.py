from __future__ import annotations

import json
from pathlib import Path

from src.eval.filmmatch_luma_preserving_chroma_adjudication import (
    adjudicate_luma_preserving_chroma,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_bl11_evidence_closes_candidate_preference() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2bl11_filmmatch_luma_preserving_chroma_transplant_v1.json"
        ).read_text(encoding="utf-8")
    )
    base = ROOT / "outputs/u5_r2bl11_filmmatch_luma_preserving_chroma_transplant_v1"
    result = adjudicate_luma_preserving_chroma(
        config=config,
        observations_path=(
            ROOT
            / "configs/u5_r2bl11_filmmatch_luma_preserving_chroma_transplant_observations_v1.json"
        ),
        run_a=base / "run_a",
        run_b=base / "run_b",
        blind_dir=base / "blind_review",
    )
    assert result["automatic_gate_pass"] is True
    assert result["confirmed_severe_artifact_count"] == 0
    assert result["candidate_choices"] == 3
    assert result["ao6_choices"] == 14
    assert result["decision"] == "close_bl11_preference_retain_ao6"
    assert result["product_integration_opened"] is False
