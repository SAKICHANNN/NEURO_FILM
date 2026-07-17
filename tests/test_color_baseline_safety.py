from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluate_render_safety import evaluate
from scripts.pipeline_color_baseline import (
    load_guardrail_config,
    load_profile_values,
    style_transfer,
    style_transfer_rgb,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rgb() -> Image.Image:
    x = np.linspace(0, 255, 96, dtype=np.uint8)
    y = np.linspace(24, 232, 64, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, 255 - xx // 2], axis=2).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_safe_rich_render_reserves_output_headroom(tmp_path: Path) -> None:
    stats = json.loads((ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8"))
    profile = load_profile_values(ROOT / "configs" / "color_rendering_profiles.yaml", "safe-rich", "velvia_50")
    image = synthetic_rgb()
    output = style_transfer(
        image,
        stats["styles"]["velvia_50"],
        "velvia_50",
        strength=profile["strength"],
        luma_strength=profile["luma_strength"],
        grain=profile["grain"],
        seed=7,
        gamut_safe=profile["gamut_safe"],
        gamut_mode=profile["gamut_mode"],
        tone_rolloff=profile["tone_rolloff"],
        shadow_floor_l=profile["shadow_floor_l"],
        highlight_ceiling_l=profile["highlight_ceiling_l"],
        preserve_luma_detail_strength=profile["preserve_luma_detail"],
        chroma_curve_strength=profile["chroma_curve_strength"],
        output_margin=profile["output_margin"],
        guardrails=load_guardrail_config(ROOT / "configs" / "color_guardrails.json", "velvia_50"),
        dither=profile["dither"],
    )
    arr = np.asarray(output, dtype=np.uint8)
    assert arr.min() >= 4
    assert arr.max() <= 251

    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    image.save(before_path, "PNG")
    output.save(after_path, "PNG")
    metrics = evaluate(before_path, after_path)
    assert metrics["clip"]["new_clipped_pixel_count"] == 0


def test_pil_wrapper_is_explicit_quantization_of_float_core() -> None:
    image = synthetic_rgb()
    stats = {"mean": [52.0, 8.0, -4.0], "std": [18.0, 14.0, 12.0]}
    kwargs = {
        "strength": 0.7,
        "luma_strength": 0.5,
        "grain": 0.0,
        "seed": 11,
        "gamut_safe": True,
        "output_margin": 4,
    }
    wrapped = np.asarray(style_transfer(image, stats, "velvia_50", **kwargs))
    floating = style_transfer_rgb(
        np.asarray(image, dtype=np.float32) / 255.0,
        stats,
        "velvia_50",
        **kwargs,
    )
    assert np.array_equal(wrapped, np.rint(floating * 255.0).astype(np.uint8))


def test_bw_styles_are_exactly_achromatic_with_guardrails_grain_and_dither() -> None:
    stats = json.loads((ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260717)
    random_rgb = rng.random((31, 47, 3), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, 47, dtype=np.float32)[None, :, None]
    neutral = np.broadcast_to(ramp, (8, 47, 3)).copy()
    primaries = np.zeros((8, 47, 3), dtype=np.float32)
    primaries[:, :16, 0] = 1.0
    primaries[:, 16:32, 1] = 1.0
    primaries[:, 32:, 2] = 1.0
    fixture = np.concatenate((random_rgb, neutral, primaries), axis=0)

    for style in ("hp5", "tri_x_400"):
        profile = load_profile_values(
            ROOT / "configs" / "color_rendering_profiles.yaml", "safe-rich", style
        )
        output = style_transfer_rgb(
            fixture,
            stats["styles"][style],
            style,
            strength=profile["strength"],
            luma_strength=profile["luma_strength"],
            grain=0.013,
            seed=20260717,
            gamut_safe=profile["gamut_safe"],
            gamut_mode=profile["gamut_mode"],
            tone_rolloff=profile["tone_rolloff"],
            shadow_floor_l=profile["shadow_floor_l"],
            highlight_ceiling_l=profile["highlight_ceiling_l"],
            preserve_luma_detail_strength=profile["preserve_luma_detail"],
            chroma_curve_strength=profile["chroma_curve_strength"],
            output_margin=profile["output_margin"],
            guardrails=load_guardrail_config(ROOT / "configs" / "color_guardrails.json", style),
            dither=profile["dither"],
        )
        assert float(np.ptp(output, axis=2).max()) <= 2e-6
        for scale, dtype in ((255.0, np.uint8), (65535.0, np.uint16)):
            quantized = np.rint(output * scale).astype(dtype)
            assert np.array_equal(quantized[..., 0], quantized[..., 1])
            assert np.array_equal(quantized[..., 1], quantized[..., 2])


def test_bw_projection_does_not_change_frozen_velvia_output() -> None:
    stats = json.loads((ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8"))
    profile = load_profile_values(
        ROOT / "configs" / "color_rendering_profiles.yaml", "safe-rich", "velvia_50"
    )
    y, x = np.mgrid[0:41, 0:53]
    fixture = np.stack((x / 52, y / 40, ((3 * x + 5 * y) % 67) / 66), axis=2).astype(np.float32)
    output = style_transfer_rgb(
        fixture,
        stats["styles"]["velvia_50"],
        "velvia_50",
        strength=profile["strength"],
        luma_strength=profile["luma_strength"],
        grain=0.013,
        seed=20260717,
        gamut_safe=profile["gamut_safe"],
        gamut_mode=profile["gamut_mode"],
        tone_rolloff=profile["tone_rolloff"],
        shadow_floor_l=profile["shadow_floor_l"],
        highlight_ceiling_l=profile["highlight_ceiling_l"],
        preserve_luma_detail_strength=profile["preserve_luma_detail"],
        chroma_curve_strength=profile["chroma_curve_strength"],
        output_margin=profile["output_margin"],
        guardrails=load_guardrail_config(ROOT / "configs" / "color_guardrails.json", "velvia_50"),
        dither=profile["dither"],
    )
    quantized = np.rint(output * 255.0).astype(np.uint8)
    assert hashlib.sha256(quantized.tobytes()).hexdigest() == (
        "72a7e30e1b3f9640ed764c8ddee40e6da028c8afda4e78dd40d88b2d27e85307"
    )
