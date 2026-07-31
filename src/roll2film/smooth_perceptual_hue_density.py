"""Smooth density, hue-response and split-tone film-look primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.color_engine.gamut import compress_chroma_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)


@dataclass(frozen=True)
class SmoothPerceptualHueDensityResponse:
    tone_power: float = 1.35
    luminance_floor: float = 0.0001
    base_chroma_gain: float = 1.3
    first_harmonic_gain: float = 0.12
    first_harmonic_center_degrees: float = 35.0
    second_harmonic_gain: float = -0.08
    second_harmonic_center_degrees: float = 110.0
    hue_warp: float = 0.11
    hue_warp_center_degrees: float = 45.0
    luminance_hue_tilt: float = 0.04
    split_tone_a: float = 4.8
    split_tone_b: float = 8.0
    gamut_iterations: int = 24
    encoded_margin: float = 1.0 / 65535.0
    row_chunk: int = 128

    def __post_init__(self) -> None:
        finite = (
            self.tone_power,
            self.luminance_floor,
            self.base_chroma_gain,
            self.first_harmonic_gain,
            self.first_harmonic_center_degrees,
            self.second_harmonic_gain,
            self.second_harmonic_center_degrees,
            self.hue_warp,
            self.hue_warp_center_degrees,
            self.luminance_hue_tilt,
            self.split_tone_a,
            self.split_tone_b,
            self.encoded_margin,
        )
        if (
            not all(np.isfinite(finite))
            or self.tone_power <= 0.0
            or not 0.0 < self.luminance_floor < 0.01
            or self.base_chroma_gain <= 0.0
            or abs(self.first_harmonic_gain) >= 0.5
            or abs(self.second_harmonic_gain) >= 0.5
            or abs(self.hue_warp) >= 0.5
            or abs(self.luminance_hue_tilt) >= 0.25
            or self.gamut_iterations <= 0
            or not 0.0 < self.encoded_margin < 0.01
            or self.row_chunk <= 0
        ):
            raise ValueError("invalid smooth perceptual hue-density response")

    def _target_lab(self, linear: np.ndarray) -> np.ndarray:
        lab = linear_rgb_to_lab(
            linear.astype(np.float32, copy=False),
            working_space="linear_srgb",
        ).astype(np.float64)
        luminance = lab[..., 0] / 100.0
        a = lab[..., 1]
        b = lab[..., 2]
        chroma = np.hypot(a, b)
        hue = np.arctan2(b, a)
        mid = 4.0 * luminance * (1.0 - luminance)

        light = np.power(luminance, self.tone_power)
        dark = np.power(
            np.maximum(1.0 - luminance, 0.0),
            self.tone_power,
        )
        density_luminance = np.clip(
            np.divide(
                light,
                light + dark,
                out=np.zeros_like(light),
                where=(light + dark) > 0.0,
            ),
            self.luminance_floor,
            1.0 - self.luminance_floor,
        )

        first_center = np.deg2rad(self.first_harmonic_center_degrees)
        second_center = np.deg2rad(self.second_harmonic_center_degrees)
        warp_center = np.deg2rad(self.hue_warp_center_degrees)
        chroma_gain = (
            self.base_chroma_gain
            + self.first_harmonic_gain * np.cos(hue - first_center)
            + self.second_harmonic_gain
            * np.cos(2.0 * (hue - second_center))
        )
        hue_shift = (
            self.hue_warp * np.sin(hue - warp_center) * mid
            + self.luminance_hue_tilt * (2.0 * luminance - 1.0)
        )
        target_hue = hue + hue_shift
        split = (2.0 * luminance - 1.0) * mid
        target = np.stack(
            (
                100.0 * density_luminance,
                chroma * chroma_gain * np.cos(target_hue)
                + self.split_tone_a * split,
                chroma * chroma_gain * np.sin(target_hue)
                + self.split_tone_b * split,
            ),
            axis=-1,
        ).astype(np.float32)
        return compress_chroma_to_working_gamut(
            target,
            working_space="linear_srgb",
            iterations=self.gamut_iterations,
            tolerance=2e-6,
        )

    def _apply_chunk(
        self, source: np.ndarray, *, strength: float
    ) -> np.ndarray:
        linear = encoded_srgb_to_linear(source).astype(np.float32)
        target_lab = self._target_lab(linear)
        candidate_linear = lab_to_linear_rgb(
            target_lab,
            working_space="linear_srgb",
        ).astype(np.float64)
        linear_source = linear.astype(np.float64)
        linear_residual = candidate_linear - linear_source
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            linear_upper = np.where(
                linear_residual > 0.0,
                (1.0 - linear_source) / linear_residual,
                np.inf,
            )
            linear_lower = np.where(
                linear_residual < 0.0,
                -linear_source / linear_residual,
                np.inf,
            )
        linear_scale = np.clip(
            np.min(
                np.minimum(linear_upper, linear_lower),
                axis=-1,
            ),
            0.0,
            1.0,
        )
        safe_linear = (
            linear_source
            + linear_residual * linear_scale[..., None]
        )
        candidate = linear_srgb_to_encoded(safe_linear)
        residual = (candidate - source) * strength
        positive = residual > 0.0
        negative = residual < 0.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            upper = np.where(
                positive,
                (1.0 - self.encoded_margin - source) / residual,
                np.inf,
            )
            lower = np.where(
                negative,
                (self.encoded_margin - source) / residual,
                np.inf,
            )
        scale = np.clip(
            np.min(np.minimum(upper, lower), axis=-1),
            0.0,
            1.0,
        )
        output = source + residual * scale[..., None]
        endpoint = np.all(source == 0.0, axis=-1) | np.all(
            source == 1.0, axis=-1
        )
        return np.where(endpoint[..., None], source, output)

    def apply(
        self, encoded_rgb: np.ndarray, *, strength: float = 1.0
    ) -> np.ndarray:
        source = np.asarray(encoded_rgb)
        amount = float(strength)
        if (
            source.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}
            or source.ndim != 3
            or source.shape[-1] != 3
            or source.size == 0
            or not np.all(np.isfinite(source))
            or np.any(source < 0.0)
            or np.any(source > 1.0)
            or not np.isfinite(amount)
            or not 0.0 <= amount <= 1.0
        ):
            raise ValueError(
                "expected finite HxWx3 encoded RGB and strength in [0,1]"
            )
        if amount == 0.0:
            return np.array(source, copy=True)
        output = np.empty_like(source)
        for y0 in range(0, source.shape[0], self.row_chunk):
            y1 = min(y0 + self.row_chunk, source.shape[0])
            output[y0:y1] = self._apply_chunk(
                source[y0:y1].astype(np.float64),
                strength=amount,
            ).astype(source.dtype)
        if (
            not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("smooth hue-density response escaped RGB")
        return output


__all__ = ["SmoothPerceptualHueDensityResponse"]
