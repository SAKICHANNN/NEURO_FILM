"""Bounded exposure-dependent log-chroma film-look primitive.

This is a generic film-inspired point operator, not an identified stock
response.  It couples a monotone luminance shoulder with a small,
luminance-dependent opponent transform, then analytically scales the complete
RGB residual to remain inside the requested cube margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


@dataclass(frozen=True)
class LogChromaFilmResponse:
    contrast: float = 1.12
    chroma_gain: float = 1.16
    midtone_chroma_lift: float = 0.08
    opponent_rotation: float = 0.07
    margin: float = 1.0 / 65535.0

    def __post_init__(self) -> None:
        values = (
            self.contrast,
            self.chroma_gain,
            self.midtone_chroma_lift,
            self.opponent_rotation,
            self.margin,
        )
        if (
            not all(np.isfinite(values))
            or self.contrast <= 0.0
            or self.chroma_gain <= 0.0
            or self.midtone_chroma_lift < 0.0
            or abs(self.opponent_rotation) >= 0.25
            or not 0.0 < self.margin < 0.01
        ):
            raise ValueError("invalid log-chroma film response")

    def apply(self, linear_rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        source = np.asarray(linear_rgb)
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
            raise ValueError("expected finite RGB in [0,1] and strength in [0,1]")
        if amount == 0.0:
            return np.array(source, copy=True)

        x = source.astype(np.float64)
        y = np.sum(x * _LUMA, axis=-1)
        eps = np.finfo(np.float64).eps

        # Symmetric power sigmoid: monotone, exact at 0/1, toe and shoulder.
        yp = np.power(y, self.contrast)
        invp = np.power(1.0 - y, self.contrast)
        y_film = np.divide(yp, yp + invp, out=np.zeros_like(y), where=(yp + invp) > 0)

        safe = np.maximum(x, eps)
        rg = np.log(safe[..., 0] / safe[..., 1])
        bg = np.log(safe[..., 2] / safe[..., 1])
        tone = np.clip(y, 0.0, 1.0)
        mid = 4.0 * tone * (1.0 - tone)
        gain = self.chroma_gain + self.midtone_chroma_lift * mid
        rotation = self.opponent_rotation * (2.0 * tone - 1.0)
        rg2 = gain * rg - rotation * bg
        bg2 = rotation * rg + gain * bg

        ratios = np.stack((np.exp(rg2), np.ones_like(rg2), np.exp(bg2)), axis=-1)
        ratio_luma = np.sum(ratios * _LUMA, axis=-1)
        candidate = ratios * np.divide(
            y_film,
            ratio_luma,
            out=np.zeros_like(y_film),
            where=ratio_luma > 0,
        )[..., None]

        residual = (candidate - x) * amount
        alpha = np.ones_like(y)
        positive = residual > 0.0
        negative = residual < 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            upper = np.where(positive, (1.0 - self.margin - x) / residual, np.inf)
            lower = np.where(negative, (self.margin - x) / residual, np.inf)
        alpha = np.minimum(alpha, np.min(np.minimum(upper, lower), axis=-1))
        alpha = np.clip(alpha, 0.0, 1.0)
        output = x + residual * alpha[..., None]
        if not np.all(np.isfinite(output)):
            raise RuntimeError("log-chroma response produced non-finite output")
        return output.astype(source.dtype)


__all__ = ["LogChromaFilmResponse"]
