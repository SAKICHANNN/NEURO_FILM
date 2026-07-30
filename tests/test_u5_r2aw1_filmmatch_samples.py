from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.filmmatch_chart_pairs import (
    HUE_LABELS,
    _hue_sector,
    _sample_rectangle,
    chart_sample_geometry,
    slog3_to_linear_reflection,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT / "configs/u5_r2aw1_filmmatch_code_domain_capacity_v1.json"
    ).read_text(encoding="utf-8")
)


def test_slog3_inverse_matches_official_reference_codes() -> None:
    encoded = np.asarray([95.0, 420.0, 598.0]) / 1023.0
    decoded = slog3_to_linear_reflection(encoded)
    # Sony publishes integer 10-bit reference codes, so the white value is
    # quantized rather than the exact floating OETF result.
    np.testing.assert_allclose(decoded, [0.0, 0.18, 0.9], atol=1e-3, rtol=0.0)


def test_chart_geometry_is_exact_and_inside_canvas() -> None:
    geometry = chart_sample_geometry(CONFIG)
    assert len(geometry) == 164
    assert sum(row["chart"] == "sg" for row in geometry) == 140
    assert sum(row["chart"] == "classic" for row in geometry) == 24
    height, width = CONFIG["sampling"]["reflective"]["canvas"]
    assert all(
        row["half_x"] <= row["center_x"] < width - row["half_x"]
        and row["half_y"] <= row["center_y"] < height - row["half_y"]
        for row in geometry
    )


def test_rectangle_sampler_returns_channel_medians() -> None:
    image = np.zeros((9, 11, 3), dtype=np.float64)
    image[3:6, 4:7] = [0.2, 0.4, 0.8]
    sampled = _sample_rectangle(
        image, center_x=5, center_y=4, half_x=1, half_y=1
    )
    np.testing.assert_array_equal(sampled, [0.2, 0.4, 0.8])


def test_hue_sector_covers_six_controlled_directions() -> None:
    def slog3(linear: np.ndarray) -> np.ndarray:
        return np.where(
            linear >= 0.01125,
            (
                420.0
                + np.log10((linear + 0.01) / 0.19) * 261.5
            )
            / 1023.0,
            (
                linear * (171.2102946929 - 95.0) / 0.01125 + 95.0
            )
            / 1023.0,
        )

    directions = (
        [0.8, 0.1, 0.1],
        [0.8, 0.8, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.8],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.8],
    )
    assert tuple(_hue_sector(slog3(np.asarray(row))) for row in directions) == HUE_LABELS


def test_code_domain_claim_ceiling_remains_non_calibrated() -> None:
    assert CONFIG["gates"]["code_domain_operator_fitting_allowed"] is True
    assert CONFIG["gates"]["calibrated_claim_allowed"] is False
    assert CONFIG["gates"]["product_integration_allowed"] is False
    assert "paired code-domain Look Approximation" in CONFIG["claim_ceiling"]
