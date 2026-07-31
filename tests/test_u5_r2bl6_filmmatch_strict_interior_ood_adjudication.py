from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.filmmatch_strict_interior_ood_adjudication import (
    adjudicate_strict_interior_ood,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/u5_r2bl6_filmmatch_strict_interior_ood_v1"


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/u5_r2bl6_filmmatch_strict_interior_ood_v1.json")
        .read_text(encoding="utf-8")
    )


def test_formal_bl6_adjudication_passes() -> None:
    result = adjudicate_strict_interior_ood(
        config=_config(),
        observations_path=ROOT
        / "configs/u5_r2bl6_filmmatch_strict_interior_ood_observations_v1.json",
        run_a=OUTPUT / "run_a",
        run_b=OUTPUT / "run_b",
    )
    assert result["candidate_round_wins"] == 3
    assert result["candidate_aggregate_choices"] == 32
    assert result["passed"] is True
    assert result["product_integration_opened"] is False


def test_adjudication_rejects_nonblind_observation(tmp_path: Path) -> None:
    source = ROOT / "configs/u5_r2bl6_filmmatch_strict_interior_ood_observations_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["revealed_mapping_at_capture"] = True
    altered = tmp_path / "observations.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not sealed"):
        adjudicate_strict_interior_ood(
            config=_config(),
            observations_path=altered,
            run_a=OUTPUT / "run_a",
            run_b=OUTPUT / "run_b",
        )
