from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline_color_baseline import (
    SafeLabSourceContext,
    _style_transfer_rgb_with_context,
    build_safe_lab_source_context,
    legacy_uniform_dither_window,
    load_guardrail_config,
    load_profile_values,
    style_transfer_rgb,
    style_transfer_rgb_with_source_context,
    style_transfer_rgb_tiled,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_CONFIG = ROOT / "configs" / "color_rendering_profiles.yaml"
STATS = json.loads((ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8"))["styles"]
GUARDRAILS = ROOT / "configs" / "color_guardrails.json"


def _frozen_kwargs() -> dict:
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


@pytest.mark.parametrize(
    ("dither", "expected_hash"),
    [
        (0.0, "3229cf98e4691964b0101c9e2b0288d9fed27e04eac84a9aa6922c0bfa8bfd0a"),
        (0.35, "10eea738bf9c673ffb017ff093699dcb8425dbb79d5e5a0f1b5650f3bb3a9d7a"),
    ],
)
def test_full_frame_context_refactor_preserves_frozen_pixels(dither, expected_hash):
    image = np.random.default_rng(20260717).random((17, 19, 3), dtype=np.float32)
    output = style_transfer_rgb(image, dither=dither, **_frozen_kwargs())
    assert hashlib.sha256(output.tobytes()).hexdigest() == expected_hash


@pytest.mark.parametrize("dither", [0.0, 0.35])
def test_public_source_context_application_matches_direct_full_frame(dither):
    image = np.random.default_rng(20260728).random((17, 19, 3), dtype=np.float32)
    kwargs = {**_frozen_kwargs(), "dither": dither}
    direct = style_transfer_rgb(image, **kwargs)
    explicit = style_transfer_rgb_with_source_context(
        image,
        **kwargs,
        source_context=build_safe_lab_source_context(image),
    )
    np.testing.assert_array_equal(explicit, direct)


def test_public_source_context_rejects_a_different_frame_shape():
    image = np.random.default_rng(23).random((17, 19, 3), dtype=np.float32)
    context = build_safe_lab_source_context(image)
    with pytest.raises(ValueError, match="same full-frame shape"):
        style_transfer_rgb_with_source_context(
            image[:16],
            **_frozen_kwargs(),
            source_context=context,
        )


@pytest.mark.parametrize("window", [(0, 9, 0, 13), (2, 7, 3, 11), (0, 2, 9, 13), (8, 9, 12, 13)])
def test_coordinate_dither_exactly_replays_legacy_sequence(window):
    shape = (9, 13, 3)
    seed = 41
    legacy = np.random.default_rng(seed + 1009).uniform(-0.5, 0.5, size=shape).astype(np.float32)
    y0, y1, x0, x1 = window
    replay = legacy_uniform_dither_window(
        seed=seed,
        full_shape=shape,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
    )
    np.testing.assert_array_equal(replay, legacy[y0:y1, x0:x1])


def test_all_safe_rich_styles_match_tiled_with_no_seams():
    image = np.random.default_rng(8675309).random((31, 47, 3), dtype=np.float32)
    style_ids = sorted(STATS)
    assert len(style_ids) == 8
    for style in style_ids:
        profile = load_profile_values(PROFILE_CONFIG, "safe-rich", style)
        kwargs = {
            "stats": STATS[style],
            "style": style,
            "strength": profile["strength"],
            "luma_strength": profile["luma_strength"],
            "grain": profile["grain"],
            "seed": 23,
            "gamut_safe": profile["gamut_safe"],
            "gamut_mode": profile["gamut_mode"],
            "tone_rolloff": profile["tone_rolloff"],
            "shadow_floor_l": profile["shadow_floor_l"],
            "highlight_ceiling_l": profile["highlight_ceiling_l"],
            "preserve_luma_detail_strength": profile["preserve_luma_detail"],
            "chroma_curve_strength": profile["chroma_curve_strength"],
            "output_margin": profile["output_margin"],
            "guardrails": load_guardrail_config(GUARDRAILS, style),
            "dither": profile["dither"],
        }
        full = style_transfer_rgb(image, **kwargs)
        tiled, metadata = style_transfer_rgb_tiled(image, tile_size=11, **kwargs)
        error = np.abs(full - tiled)
        assert float(error.max()) <= 1e-6, style
        seam_mask = np.zeros(image.shape[:2], dtype=bool)
        seam_mask[10::11, :] = True
        seam_mask[:, 10::11] = True
        assert float(error[seam_mask].max()) <= 1e-6, style
        assert metadata.halo == (5 if profile["preserve_luma_detail"] > 0 else 0)
        assert metadata.max_expanded_shape[0] <= 11 + 2 * metadata.halo
        assert metadata.max_expanded_shape[1] <= 11 + 2 * metadata.halo


def test_tiled_safe_lab_repeats_byte_identically():
    image = np.random.default_rng(7).random((23, 29, 3), dtype=np.float32)
    kwargs = {**_frozen_kwargs(), "dither": 0.35, "tile_size": 8}
    first, first_metadata = style_transfer_rgb_tiled(image, **kwargs)
    second, second_metadata = style_transfer_rgb_tiled(image, **kwargs)
    assert first.tobytes() == second.tobytes()
    assert first_metadata == second_metadata


def test_tiled_safe_lab_rejects_nonzero_legacy_grain():
    image = np.zeros((7, 9, 3), dtype=np.float32)
    kwargs = {**_frozen_kwargs(), "grain": 0.01, "dither": 0.0, "tile_size": 4}
    with pytest.raises(ValueError, match="does not support legacy colour-core grain"):
        style_transfer_rgb_tiled(image, **kwargs)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, source_shape=(0, 7, 3)),
        lambda context: replace(context, pixel_count=context.pixel_count + 1),
        lambda context: replace(context, lab_mean=(float("nan"), 0.0, 0.0)),
        lambda context: replace(context, lab_std=(0.0, 1.0, 1.0)),
    ],
)
def test_invalid_source_context_fails_closed(mutation):
    image = np.zeros((5, 7, 3), dtype=np.float32)
    context = mutation(build_safe_lab_source_context(image))
    with pytest.raises(ValueError, match="source_context"):
        _style_transfer_rgb_with_context(
            image,
            {"mean": [50.0, 0.0, 0.0], "std": [20.0, 10.0, 10.0]},
            "velvia_50",
            0.5,
            0.5,
            0.0,
            1,
            True,
            source_context=context,
        )


def test_dither_window_outside_full_shape_fails_closed():
    with pytest.raises(ValueError, match="outside"):
        legacy_uniform_dither_window(seed=1, full_shape=(5, 7, 3), y0=0, y1=6, x0=0, x1=7)


def test_context_constructor_rejects_empty_or_nonfinite_rgb():
    with pytest.raises(ValueError):
        build_safe_lab_source_context(np.zeros((0, 7, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        build_safe_lab_source_context(np.full((5, 7, 3), np.inf, dtype=np.float32))
    with pytest.raises(ValueError):
        build_safe_lab_source_context(np.zeros((5, 7), dtype=np.float32))
