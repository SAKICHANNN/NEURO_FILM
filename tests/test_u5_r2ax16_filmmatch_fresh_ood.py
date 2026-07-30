import json
from pathlib import Path

import numpy as np

from src.eval.filmmatch_chart_pairs import (
    linear_reflection_to_slog3,
    slog3_to_linear_reflection,
)
from src.eval.filmmatch_fresh_ood import (
    display_srgb_to_sgamut3_cine_slog3,
    linear_srgb_to_sgamut3_cine,
)


def test_slog3_forward_matches_sony_reference_codes_and_roundtrips() -> None:
    reflection = np.asarray([0.0, 0.18, 0.9])
    encoded = linear_reflection_to_slog3(reflection)
    np.testing.assert_allclose(
        encoded * 1023.0, [95.0, 420.0, 598.0], atol=0.5
    )
    np.testing.assert_allclose(
        slog3_to_linear_reflection(encoded), reflection, atol=1e-12
    )


def test_srgb_cube_is_contained_by_sgamut3_cine_adapter() -> None:
    cube = np.stack(
        np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    converted = linear_srgb_to_sgamut3_cine(cube)
    assert np.min(converted) >= -1e-12
    assert np.max(converted) <= 1.0 + 1e-12
    code = display_srgb_to_sgamut3_cine_slog3(cube)
    assert np.all(np.isfinite(code))
    assert np.min(code) >= 0.0
    assert np.max(code) <= 1.0


def test_ax16_freezes_population_and_forbids_refit() -> None:
    config = json.loads(
        Path("configs/u5_r2ax16_filmmatch_fresh_ood_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["population"]["fixed_before_filmmatch_operator"] is True
    assert config["population"]["operator_fitting_allowed"] is False
    assert config["gates"]["operator_refit_allowed"] is False
    assert config["gates"]["population_replacement_allowed"] is False
