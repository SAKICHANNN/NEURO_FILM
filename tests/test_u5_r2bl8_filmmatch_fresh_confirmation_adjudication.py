from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.filmmatch_fresh_confirmation_adjudication import (
    adjudicate_fresh_confirmation,
)


ROOT = Path(__file__).resolve().parents[1]


def _adjudicate(observations_path: Path):
    config = json.loads(
        (ROOT / "configs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_v1.json").read_text(encoding="utf-8")
    )
    bl8 = ROOT / "outputs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_v1"
    bl9 = ROOT / "outputs/u5_r2bl9_filmmatch_fresh_nonbasic_audit_v1"
    return adjudicate_fresh_confirmation(
        config=config,
        observations_path=observations_path,
        run_a=bl8 / "run_a",
        run_b=bl8 / "run_b",
        nonbasic_a_path=bl9 / "run_a.json",
        nonbasic_b_path=bl9 / "run_b.json",
    )


def test_formal_bl8_adjudication_retains_ao6() -> None:
    result = _adjudicate(
        ROOT / "configs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_observations_v1.json"
    )
    assert result["candidate_round_wins"] == 0
    assert result["candidate_aggregate_choices"] == 16
    assert result["candidate_source_majority_wins"] == 5
    assert result["preference_passed"] is False
    assert result["nonbasic_passed"] is True
    assert result["decision"] == "retain_ao6_incumbent_close_bl5_preference_promotion"


def test_adjudication_rejects_nonblind_observation(tmp_path: Path) -> None:
    source = ROOT / "configs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_observations_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["revealed_mapping_at_capture"] = True
    altered = tmp_path / "observations.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not sealed"):
        _adjudicate(altered)
