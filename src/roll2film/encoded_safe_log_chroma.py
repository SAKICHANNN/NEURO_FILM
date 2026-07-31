"""Encoded-domain-safe nonlinear log-chroma film-look primitive.

The candidate is formed from scene-linear luminance and opponent ratios, but
the complete residual is analytically bounded in display-encoded RGB.  This
keeps final integer-code safety aligned with the actual output contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


def _encoded_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    )


def _linear_to_encoded(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(np.maximum(value, 0.0), 1.0 / 2.4) - 0.055,
    )


@dataclass(frozen=True)
class EncodedSafeLogChromaFilmResponse:
    contrast: float = 1.18
    red_green_gain: float = 1.22
    blue_yellow_gain: float = 1.14
    midtone_chroma_lift: float = 0.12
    opponent_rotation: float = 0.09
    hue_bend: float = 0.06
    encoded_margin: float = 1.0 / 65535.0

    def __post_init__(self) -> None:
        values = (
            self.contrast,
            self.red_green_gain,
            self.blue_yellow_gain,
            self.midtone_chroma_lift,
            self.opponent_rotation,
            self.hue_bend,
            self.encoded_margin,
        )
        if (
            not all(np.isfinite(values))
            or self.contrast <= 0.0
            or self.red_green_gain <= 0.0
            or self.blue_yellow_gain <= 0.0
            or self.midtone_chroma_lift < 0.0
            or abs(self.opponent_rotation) >= 0.25
            or abs(self.hue_bend) >= 0.25
            or not 0.0 < self.encoded_margin < 0.01
        ):
            raise ValueError("invalid encoded-safe log-chroma response")

    def apply(
        self, encoded_rgb: np.ndarray, *, strength: float = 1.0
    ) -> np.ndarray:
        source = np.asarray(encoded_rgb)
        amount = float(strength)
        if (
            source.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}
            or source.ndim < 2
            or source.shape[-1] != 3
            or source.size == 0
            or not np.all(np.isfinite(source))
            or np.any(source < 0.0)
            or np.any(source > 1.0)
            or not np.isfinite(amount)
            or not 0.0 <= amount <= 1.0
        ):
            raise ValueError(
                "expected finite encoded RGB in [0,1] and strength in [0,1]"
            )
        if amount == 0.0:
            return np.array(source, copy=True)

        encoded = source.astype(np.float64)
        linear = _encoded_to_linear(encoded)
        y = np.sum(linear * _LUMA, axis=-1)
        eps = np.finfo(np.float64).tiny

        yp = np.power(y, self.contrast)
        invp = np.power(np.maximum(1.0 - y, 0.0), self.contrast)
        y_film = np.divide(
            yp,
            yp + invp,
            out=np.zeros_like(y),
            where=(yp + invp) > 0,
        )

        safe = np.maximum(linear, eps)
        rg = np.log(safe[..., 0] / safe[..., 1])
        bg = np.log(safe[..., 2] / safe[..., 1])
        mid = 4.0 * np.clip(y, 0.0, 1.0) * (
            1.0 - np.clip(y, 0.0, 1.0)
        )
        rg_gain = self.red_green_gain + self.midtone_chroma_lift * mid
        bg_gain = self.blue_yellow_gain + 0.75 * self.midtone_chroma_lift * mid
        rotation = self.opponent_rotation * (2.0 * y - 1.0)
        rg2 = (
            rg_gain * rg
            - rotation * bg
            + self.hue_bend * mid * np.tanh(bg)
        )
        bg2 = (
            rotation * rg
            + bg_gain * bg
            - self.hue_bend * mid * np.tanh(rg)
        )
        log_ratios = np.stack(
            (rg2, np.zeros_like(rg2), bg2), axis=-1
        )
        # This is algebraically the same luminance normalization as direct
        # ratio exponentiation, but stays finite when one encoded channel is
        # exactly zero and its opponent is not.
        log_scale = np.max(log_ratios, axis=-1, keepdims=True)
        scaled_ratios = np.exp(log_ratios - log_scale)
        ratio_luma = np.sum(scaled_ratios * _LUMA, axis=-1)
        linear_candidate = scaled_ratios * np.divide(
            y_film,
            ratio_luma,
            out=np.zeros_like(y_film),
            where=ratio_luma > 0,
        )[..., None]
        encoded_candidate = _linear_to_encoded(linear_candidate)

        residual = (encoded_candidate - encoded) * amount
        positive = residual > 0.0
        negative = residual < 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            upper = np.where(
                positive,
                (1.0 - self.encoded_margin - encoded) / residual,
                np.inf,
            )
            lower = np.where(
                negative,
                (self.encoded_margin - encoded) / residual,
                np.inf,
            )
        alpha = np.clip(
            np.min(np.minimum(upper, lower), axis=-1), 0.0, 1.0
        )
        output = encoded + residual * alpha[..., None]
        endpoint = np.all(encoded == 0.0, axis=-1) | np.all(
            encoded == 1.0, axis=-1
        )
        output = np.where(endpoint[..., None], encoded, output)
        if (
            not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError(
                "encoded-safe log-chroma response escaped display RGB"
            )
        return output.astype(source.dtype)


__all__ = ["EncodedSafeLogChromaFilmResponse"]
