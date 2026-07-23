from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.density_witness_frontier import (
    DensityFrontierError,
    candidate_bank,
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
    shortlist_candidates,
)


def test_frontier_script_normalizes_explicit_relative_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_u5_r2e1_density_witness_frontier.py"
    ).read_text(encoding="utf-8")
    assert "if not manifest.is_absolute():" in source
    assert "manifest = ROOT / manifest" in source
    assert "if not output.is_absolute():" in source
    assert "output = ROOT / output" in source


def test_srgb_roundtrip_is_exact_at_codes_within_float_tolerance() -> None:
    codes = np.arange(256, dtype=np.float64) / 255.0
    first, second = np.meshgrid(codes[::31], codes[::29], indexing="ij")
    encoded = np.stack((first, second, 0.5 * (first + second)), axis=-1)
    linear = encoded_srgb_to_linear(encoded)
    replay = linear_srgb_to_encoded(linear)
    np.testing.assert_allclose(replay, encoded, atol=2e-15, rtol=0.0)


def test_transfer_helpers_fail_closed() -> None:
    with pytest.raises(DensityFrontierError, match=r"\[0, 1\]"):
        encoded_srgb_to_linear(np.full((2, 2, 3), 1.1))
    with pytest.raises(DensityFrontierError, match="finite"):
        linear_srgb_to_encoded(np.full((2, 2, 3), np.nan))


def test_candidate_bank_is_fixed_cross_product() -> None:
    config = {
        "witness_ids": ["a", "b"],
        "strengths": [0.2, 0.5],
        "candidate_id_format": "{witness}__s{strength_percent:02d}",
        "candidate_count": 4,
    }
    assert candidate_bank(config) == [
        {"candidate_id": "a__s20", "witness_id": "a", "strength": 0.2},
        {"candidate_id": "a__s50", "witness_id": "a", "strength": 0.5},
        {"candidate_id": "b__s20", "witness_id": "b", "strength": 0.2},
        {"candidate_id": "b__s50", "witness_id": "b", "strength": 0.5},
    ]


def test_shortlist_keeps_one_strength_and_three_witnesses() -> None:
    summaries = {
        "a20": {
            "witness_id": "a",
            "strength": 0.2,
            "automatic_survivor": True,
            "gold_median_style_delta_e76": 8.0,
            "gold_median_non_basic_residual_delta_e76": 7.0,
        },
        "a50": {
            "witness_id": "a",
            "strength": 0.5,
            "automatic_survivor": True,
            "gold_median_style_delta_e76": 12.0,
            "gold_median_non_basic_residual_delta_e76": 8.0,
        },
        "b50": {
            "witness_id": "b",
            "strength": 0.5,
            "automatic_survivor": True,
            "gold_median_style_delta_e76": 11.0,
            "gold_median_non_basic_residual_delta_e76": 10.0,
        },
        "c50": {
            "witness_id": "c",
            "strength": 0.5,
            "automatic_survivor": True,
            "gold_median_style_delta_e76": 10.0,
            "gold_median_non_basic_residual_delta_e76": 9.0,
        },
        "d50": {
            "witness_id": "d",
            "strength": 0.5,
            "automatic_survivor": True,
            "gold_median_style_delta_e76": 9.0,
            "gold_median_non_basic_residual_delta_e76": 6.0,
        },
    }
    config = {"shortlist": {"maximum_candidates": 3}}
    assert shortlist_candidates(summaries, config) == ["b50", "c50", "a50"]


def test_shortlist_tie_uses_lower_strength() -> None:
    summaries = {
        "a20": {
            "witness_id": "a",
            "strength": 0.2,
            "automatic_survivor": True,
            "gold_median_style_delta_e76": 8.0,
            "gold_median_non_basic_residual_delta_e76": 7.0,
        },
        "a50": {
            "witness_id": "a",
            "strength": 0.5,
            "automatic_survivor": True,
            "gold_median_style_delta_e76": 8.0,
            "gold_median_non_basic_residual_delta_e76": 100.0,
        },
    }
    assert shortlist_candidates(
        summaries,
        {"shortlist": {"maximum_candidates": 3}},
    ) == ["a20"]
