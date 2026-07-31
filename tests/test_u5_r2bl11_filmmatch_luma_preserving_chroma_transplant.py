from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.filmmatch_luma_preserving_chroma_transplant import (
    compose_luma_preserving_chroma,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (
            ROOT
            / "configs/u5_r2bl11_filmmatch_luma_preserving_chroma_transplant_v1.json"
        ).read_text(encoding="utf-8")
    )


def _compose(base: np.ndarray, donor: np.ndarray):
    candidate = _config()["candidate"]
    return compose_luma_preserving_chroma(
        base,
        donor,
        hard_boundary_epsilon_encoded_srgb=candidate[
            "hard_boundary_epsilon_encoded_srgb"
        ],
        guard_boundary_epsilon_encoded_srgb=candidate[
            "guard_boundary_epsilon_encoded_srgb"
        ],
    )


def test_contract_is_frozen_and_bound_to_bl8() -> None:
    report, decision = validate_contract(ROOT, _config())
    assert len(report["rows"]) == 34
    assert decision["decision"] == "retain_ao6_incumbent_close_bl5_preference_promotion"


def test_in_gamut_transplant_preserves_lightness_and_changes_chroma() -> None:
    base = np.full((3, 4, 3), [0.45, 0.45, 0.45], dtype=np.float64)
    donor = np.full((3, 4, 3), [0.55, 0.35, 0.42], dtype=np.float64)
    result = _compose(base, donor)
    assert np.all(result.residual_scale > 0.999999)
    assert np.max(np.abs(result.output_lab[..., 0] - result.base_lab[..., 0])) < 1e-3
    assert np.mean(np.abs(result.output - base)) > 0.01
    assert np.all((result.output >= 0.0) & (result.output <= 1.0))


def test_out_of_gamut_target_is_analytically_limited_without_clipping() -> None:
    base = np.full((4, 5, 3), [0.95, 0.05, 0.05], dtype=np.float64)
    donor = np.full((4, 5, 3), [0.0, 1.0, 1.0], dtype=np.float64)
    result = _compose(base, donor)
    assert np.any(result.residual_scale < 1.0)
    assert np.all(np.isfinite(result.output))
    assert np.all((result.output >= 0.0) & (result.output <= 1.0))


@pytest.mark.parametrize(
    "base,donor",
    [
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.zeros((2, 2, 3)), np.zeros((3, 2, 3))),
        (np.full((2, 2, 3), np.nan), np.zeros((2, 2, 3))),
        (np.full((2, 2, 3), 1.1), np.zeros((2, 2, 3))),
    ],
)
def test_invalid_inputs_fail_closed(base: np.ndarray, donor: np.ndarray) -> None:
    with pytest.raises(ValueError, match="matching finite encoded RGB"):
        _compose(base, donor)
