from __future__ import annotations

import numpy as np
import pytest

from scripts.pipeline_color_baseline import (
    build_safe_lab_source_context,
    style_transfer_rgb,
)
from src.color_engine.safe_lab_rgb_context import (
    style_transfer_rgb_with_source_context,
)


def _kwargs() -> dict:
    return {
        "stats": {"mean": [55.0, 5.0, -4.0], "std": [20.0, 18.0, 16.0]},
        "style": "velvia_50",
        "strength": 0.72,
        "luma_strength": 0.41,
        "grain": 0.0,
        "seed": 37,
        "gamut_safe": True,
        "gamut_mode": "source",
        "tone_rolloff": 0.3,
        "shadow_floor_l": 2.0,
        "highlight_ceiling_l": 97.0,
        "preserve_luma_detail_strength": 0.85,
        "chroma_curve_strength": 0.25,
        "output_margin": 4,
        "guardrails": {
            "neutral_protect": 0.3,
            "skin_protect": 0.25,
            "max_chroma_gain": 1.5,
            "max_chroma_boost": 12.0,
            "max_chroma_absolute": 68.0,
        },
    }


@pytest.mark.parametrize("dither", [0.0, 0.35])
def test_explicit_source_context_matches_direct_full_frame(dither: float) -> None:
    image = np.random.default_rng(20260728).random((17, 19, 3), dtype=np.float32)
    kwargs = {**_kwargs(), "dither": dither}
    direct = style_transfer_rgb(image, **kwargs)
    explicit = style_transfer_rgb_with_source_context(
        image,
        **kwargs,
        source_context=build_safe_lab_source_context(image),
    )
    np.testing.assert_array_equal(explicit, direct)


def test_explicit_source_context_rejects_a_different_frame_shape() -> None:
    image = np.random.default_rng(23).random((17, 19, 3), dtype=np.float32)
    context = build_safe_lab_source_context(image)
    with pytest.raises(ValueError, match="same full-frame shape"):
        style_transfer_rgb_with_source_context(
            image[:16],
            **_kwargs(),
            source_context=context,
        )
