from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u6_4a_creative_diffusion_representation import run
from src.filmfx.creative_diffusion import (
    CreativeDiffusionProfile,
    apply_creative_diffusion_linear,
    validate_creative_diffusion_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_4a_creative_diffusion_representation_v1.json"


def _profile() -> CreativeDiffusionProfile:
    return CreativeDiffusionProfile(
        profile_id="test",
        sigmas_px=(0.8, 2.0),
        scatter_fractions=(0.04, 0.06),
    )


def test_representation_report_passes_and_repeats() -> None:
    first = run(CONFIG)
    second = run(CONFIG)
    assert first == second
    assert first["passed"] is True
    assert all(first["checks"].values())


def test_operator_preserves_constant_neutral_and_bounds() -> None:
    profile = _profile()
    image = np.full((17, 19, 3), np.float32(0.4), dtype=np.float32)
    output = apply_creative_diffusion_linear(image, profile)
    np.testing.assert_allclose(output, image, rtol=0.0, atol=2e-7)

    ramp = np.linspace(0.0, 1.5, 19, dtype=np.float32)
    neutral = np.repeat(ramp[None, :, None], 17, axis=0)
    neutral = np.repeat(neutral, 3, axis=2)
    output = apply_creative_diffusion_linear(neutral, profile)
    np.testing.assert_array_equal(output[..., 0], output[..., 1])
    np.testing.assert_array_equal(output[..., 1], output[..., 2])
    assert float(output.min()) >= float(neutral.min())
    assert float(output.max()) <= float(neutral.max())


@pytest.mark.parametrize(
    "profile",
    [
        CreativeDiffusionProfile("", (1.0,), (0.1,)),
        CreativeDiffusionProfile("bad", (1.0, 0.5), (0.1, 0.1)),
        CreativeDiffusionProfile("bad", (1.0,), (-0.1,)),
        CreativeDiffusionProfile("bad", (1.0,), (0.36,)),
        CreativeDiffusionProfile("bad", (1.0,), (0.1, 0.1)),
    ],
)
def test_profile_validation_fails_closed(
    profile: CreativeDiffusionProfile,
) -> None:
    with pytest.raises(ValueError):
        validate_creative_diffusion_profile(profile)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((7, 9, 3), dtype=np.float64),
        np.zeros((7, 9), dtype=np.float32),
        np.zeros((1, 9, 3), dtype=np.float32),
        np.full((7, 9, 3), np.nan, dtype=np.float32),
        np.full((7, 9, 3), -0.1, dtype=np.float32),
    ],
)
def test_input_validation_fails_closed(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_creative_diffusion_linear(image, _profile())


def test_config_explicitly_separates_diffusion_from_film_halation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "film halation" in config["forbidden_claims"]
    assert config["source_method"]["dataset_use"] == "none"
