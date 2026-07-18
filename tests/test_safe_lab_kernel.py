from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.color_engine import (
    SafeLabSourceContext,
    apply_safe_lab_transform,
    safe_lab_context_from_lab,
)


def _kwargs(context: SafeLabSourceContext) -> dict:
    return {
        "source_context": context,
        "destination_mean": np.asarray([58.0, 7.0, -5.0], dtype=np.float32),
        "destination_std": np.asarray([19.0, 17.0, 15.0], dtype=np.float32),
        "style": "velvia_50",
        "strength": 0.73,
        "luma_strength": 0.42,
        "tone_rolloff": 0.25,
        "shadow_floor_l": 2.0,
        "highlight_ceiling_l": 98.0,
        "preserve_luma_detail_strength": 0.8,
        "chroma_curve_strength": 0.3,
        "neutral_protect": 0.25,
        "skin_protect": 0.2,
        "max_chroma_gain": 1.5,
        "max_chroma_boost": 12.0,
        "max_chroma_absolute": 68.0,
    }


def test_safe_lab_kernel_is_deterministic_and_does_not_mutate_source() -> None:
    source = np.random.default_rng(14034).normal(
        [54.0, 3.0, -2.0], [18.0, 22.0, 20.0], size=(13, 17, 3)
    ).astype(np.float32)
    frozen_source = source.copy()
    context = safe_lab_context_from_lab(source)

    first = apply_safe_lab_transform(source, **_kwargs(context))
    second = apply_safe_lab_transform(source, **_kwargs(context))

    np.testing.assert_array_equal(source, frozen_source)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()


def test_safe_lab_kernel_accepts_a_tile_with_full_image_context() -> None:
    source = np.random.default_rng(14035).normal(
        [50.0, 2.0, 1.0], [15.0, 18.0, 16.0], size=(15, 19, 3)
    ).astype(np.float32)
    context = safe_lab_context_from_lab(source)
    tile = source[2:12, 3:16].copy()

    output = apply_safe_lab_transform(tile, **_kwargs(context))

    assert output.shape == tile.shape
    assert np.isfinite(output).all()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, pixel_count=context.pixel_count + 1),
        lambda context: replace(context, lab_std=(0.0, 1.0, 1.0)),
    ],
)
def test_safe_lab_kernel_rejects_invalid_context(mutation) -> None:
    source = np.full((5, 7, 3), [50.0, 0.0, 0.0], dtype=np.float32)
    context = mutation(safe_lab_context_from_lab(source))
    with pytest.raises(ValueError, match="source_context"):
        apply_safe_lab_transform(source, **_kwargs(context))


@pytest.mark.parametrize(
    "destination_std",
    [
        np.asarray([20.0, 0.0, 10.0], dtype=np.float32),
        np.asarray([20.0, np.nan, 10.0], dtype=np.float32),
        np.asarray([20.0, 10.0], dtype=np.float32),
    ],
)
def test_safe_lab_kernel_rejects_invalid_destination_stats(destination_std: np.ndarray) -> None:
    source = np.full((5, 7, 3), [50.0, 0.0, 0.0], dtype=np.float32)
    context = safe_lab_context_from_lab(source)
    kwargs = _kwargs(context)
    kwargs["destination_std"] = destination_std
    with pytest.raises(ValueError, match="destination Lab statistics"):
        apply_safe_lab_transform(source, **kwargs)
