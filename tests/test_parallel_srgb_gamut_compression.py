from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import (
    compress_to_srgb_gamut,
    compress_to_srgb_gamut_parallel,
    load_guardrail_config,
)
from src.inference import load_render_profile, render_resolved_safe_lab_rgb

ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260826)
    source_rgb = rng.random((97, 131, 3), dtype=np.float32)
    source_lab = np.ascontiguousarray(rgb2lab(source_rgb).astype(np.float32))
    target_lab = np.ascontiguousarray(
        source_lab
        + rng.normal(0.0, (20.0, 40.0, 40.0), source_lab.shape).astype(
            np.float32
        )
    )
    return source_lab, target_lab


@pytest.mark.parametrize("workers", [1, 2, 4, 8, 128])
def test_parallel_compression_is_bit_exact(workers: int) -> None:
    source_lab, target_lab = _fixture()
    expected = compress_to_srgb_gamut(source_lab, target_lab)
    actual = compress_to_srgb_gamut_parallel(
        source_lab, target_lab, workers=workers
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_parallel_compression_rejects_invalid_workers(workers: object) -> None:
    source_lab, target_lab = _fixture()
    with pytest.raises(ValueError, match="workers"):
        compress_to_srgb_gamut_parallel(
            source_lab, target_lab, workers=workers  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("style", ["velvia_50", "portra_400", "ektar_100"])
def test_three_stock_engine_is_bit_exact_with_parallel_gamut(style: str) -> None:
    source = np.random.default_rng(20260826).random((103, 137, 3), dtype=np.float32)
    profile = load_render_profile(
        ROOT / "configs" / "render_profiles" / "safe_rich_v1.json", root=ROOT
    )
    statistics = json.loads(
        (ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8")
    )["styles"][style]
    guardrails = load_guardrail_config(
        ROOT / "configs" / "color_guardrails.json", style
    )
    kwargs = {
        "style": style,
        "style_statistics": statistics,
        "style_parameters": profile["style_parameters"][style],
        "guardrails": guardrails,
        "seed": 1729,
    }
    serial = render_resolved_safe_lab_rgb(source, **kwargs, gamut_workers=1)
    parallel = render_resolved_safe_lab_rgb(source, **kwargs, gamut_workers=8)
    assert np.array_equal(parallel, serial)
