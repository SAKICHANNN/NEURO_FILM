from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.color_engine import apply_rec2020_safe_lab
from src.color_engine.rec2020_safe_lab import REC2020_SAFE_LAB_STYLES
from src.preprocess import DecodeWarning, SourceProfile, WorkingImage, convert_linear_rgb


ROOT = Path(__file__).resolve().parents[1]
STATS = json.loads((ROOT / "configs" / "film_color_stats.json").read_text(encoding="utf-8"))["styles"]
GUARDRAILS = json.loads((ROOT / "configs" / "color_guardrails.json").read_text(encoding="utf-8"))


def _working(pixels: np.ndarray, *, working_space: str = "linear_rec2020", transfer_state: str = "display_linear") -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space=working_space,
        transfer_state=transfer_state,
        source_transfer_state="display_referred",
        source_profile=SourceProfile("cicp", "BT.2020 SDR test profile", 4),
        hdr_metadata={"nested": {"value": 7}},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=ROOT / "synthetic-rec2020.png",
        warnings=[DecodeWarning("fixture", "synthetic test input")],
    )


def _field() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 16, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, 12, dtype=np.float32)[:, None]
    red = np.broadcast_to(0.02 + 0.68 * x, (12, 16))
    green = np.broadcast_to(0.20 + 0.80 * y, (12, 16))
    blue = 0.01 + 0.49 * (1.0 - x) * (0.35 + 0.65 * y)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def _style_kwargs(style: str, gamut_mode: str) -> dict:
    stats = STATS[style]
    guardrails = dict(GUARDRAILS["defaults"])
    guardrails.update(GUARDRAILS["styles"].get(style, {}))
    return {
        "destination_mean": np.asarray(stats["mean"], dtype=np.float32),
        "destination_std": np.asarray(stats["std"], dtype=np.float32),
        "style": style,
        "strength": 0.35 if style != "velvia_50" else 0.35,
        "luma_strength": 0.02,
        "gamut_mode": gamut_mode,
        "tone_rolloff": 0.04,
        "shadow_floor_l": 1.0,
        "highlight_ceiling_l": 99.0,
        "preserve_luma_detail_strength": 0.90,
        "chroma_curve_strength": 0.45,
        "neutral_protect": guardrails["neutral_protect"],
        "skin_protect": guardrails["skin_protect"],
        "max_chroma_gain": guardrails.get("max_chroma_gain"),
        "max_chroma_boost": guardrails.get("max_chroma_boost"),
        "max_chroma_absolute": guardrails.get("max_chroma_absolute"),
    }


@pytest.mark.parametrize("style", sorted(REC2020_SAFE_LAB_STYLES))
@pytest.mark.parametrize("gamut_mode", ["source", "chroma"])
def test_all_enabled_styles_are_finite_and_in_rec2020(style: str, gamut_mode: str) -> None:
    result = apply_rec2020_safe_lab(_working(_field()), **_style_kwargs(style, gamut_mode))
    assert result.pixels.dtype == np.float32
    assert np.isfinite(result.pixels).all()
    assert float(result.pixels.min()) >= 0.0
    assert float(result.pixels.max()) <= 1.0


def test_preregistered_velvia_witness_remains_materially_outside_srgb() -> None:
    result = apply_rec2020_safe_lab(_working(_field()), **_style_kwargs("velvia_50", "source"))
    linear_srgb = convert_linear_rgb(
        result.pixels,
        source_space="linear_rec2020",
        destination_space="linear_srgb",
    )
    excursion = max(float(np.max(linear_srgb - 1.0)), float(np.max(-linear_srgb)))
    assert excursion >= 0.02


def test_adapter_is_deterministic_nonmutating_and_preserves_provenance() -> None:
    working = _working(_field())
    frozen_pixels = working.pixels.copy()
    first = apply_rec2020_safe_lab(working, **_style_kwargs("ektar_100", "source"))
    second = apply_rec2020_safe_lab(working, **_style_kwargs("ektar_100", "source"))

    np.testing.assert_array_equal(first.pixels, second.pixels)
    np.testing.assert_array_equal(working.pixels, frozen_pixels)
    assert first.working_space == working.working_space
    assert first.transfer_state == working.transfer_state
    assert first.source_transfer_state == working.source_transfer_state
    assert first.source_profile == working.source_profile
    assert first.orientation_applied == working.orientation_applied
    assert first.alpha_policy == working.alpha_policy
    assert first.bit_depth_in == working.bit_depth_in
    assert first.source_path == working.source_path
    assert first.hdr_metadata == working.hdr_metadata
    assert first.hdr_metadata is not working.hdr_metadata
    assert first.hdr_metadata["nested"] is not working.hdr_metadata["nested"]
    assert first.warnings[:-1] == working.warnings
    assert first.warnings is not working.warnings
    assert first.warnings[-1].code == "rec2020_safe_lab_research"


@pytest.mark.parametrize(
    ("working_space", "transfer_state", "style", "gamut_mode", "error"),
    [
        ("linear_srgb", "display_linear", "ektar_100", "source", "linear_rec2020"),
        ("linear_rec2020", "scene_linear", "ektar_100", "source", "display_linear"),
        ("linear_rec2020", "display_linear", "hp5", "source", "not enabled"),
        ("linear_rec2020", "display_linear", "ektar_100", "off", "source or chroma"),
    ],
)
def test_adapter_rejects_incompatible_boundaries(
    working_space: str,
    transfer_state: str,
    style: str,
    gamut_mode: str,
    error: str,
) -> None:
    working = _working(_field(), working_space=working_space, transfer_state=transfer_state)
    kwargs = _style_kwargs("ektar_100", "source")
    kwargs.update(style=style, gamut_mode=gamut_mode)
    with pytest.raises(ValueError, match=error):
        apply_rec2020_safe_lab(working, **kwargs)


def test_adapter_rejects_materially_out_of_gamut_input() -> None:
    pixels = _field()
    pixels[0, 0, 1] = 1.01
    with pytest.raises(ValueError, match="outside the working gamut"):
        apply_rec2020_safe_lab(_working(pixels), **_style_kwargs("ektar_100", "source"))
