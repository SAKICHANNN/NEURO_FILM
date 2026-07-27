from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.shared_resource_diffusion import make_patterns, simulate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u5_r2ac1_shared_resource_diffusion_v1.json").read_text(
        encoding="utf-8"
    )
)


def test_zero_strength_is_exact_identity() -> None:
    image = make_patterns(CONFIG)["square_17"]
    np.testing.assert_array_equal(
        simulate(image, CONFIG["solver"], strength=0.0)["output"], image
    )


def test_constant_neutral_stays_spatially_and_chromatically_constant() -> None:
    image = make_patterns(CONFIG)["constant_0.5"]
    output = simulate(image, CONFIG["solver"])["output"]
    assert np.ptp(output) <= 1e-12


def test_rotation_and_repeat_are_exact() -> None:
    image = make_patterns(CONFIG)["square_17"]
    first = simulate(image, CONFIG["solver"])["output"]
    second = simulate(image, CONFIG["solver"])["output"]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        simulate(np.rot90(image), CONFIG["solver"])["output"],
        np.rot90(first),
        rtol=0.0,
        atol=1e-12,
    )


def test_single_channel_activation_does_not_create_other_channels() -> None:
    image = make_patterns(CONFIG)["isolated_0"]
    output = simulate(image, CONFIG["solver"])["output"]
    assert np.max(np.abs(output[:, :, 1:])) == 0.0
