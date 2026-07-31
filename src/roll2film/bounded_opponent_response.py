"""Division-free encoded opponent film-look response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


@dataclass(frozen=True)
class BoundedOpponentFilmResponse:
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
            raise ValueError("invalid bounded opponent response")

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
        y = np.sum(encoded * _LUMA, axis=-1)
        yp = np.power(y, self.contrast)
        invp = np.power(np.maximum(1.0 - y, 0.0), self.contrast)
        y_film = np.divide(
            yp,
            yp + invp,
            out=np.zeros_like(y),
            where=(yp + invp) > 0,
        )

        rg = encoded[..., 0] - encoded[..., 1]
        bg = encoded[..., 2] - encoded[..., 1]
        mid = 4.0 * y * (1.0 - y)
        rotation = self.opponent_rotation * (2.0 * y - 1.0)
        rg2 = (
            (self.red_green_gain + self.midtone_chroma_lift * mid) * rg
            - rotation * bg
            + 0.5 * self.hue_bend * mid * np.tanh(2.0 * bg)
        )
        bg2 = (
            rotation * rg
            + (
                self.blue_yellow_gain
                + 0.75 * self.midtone_chroma_lift * mid
            )
            * bg
            - 0.5 * self.hue_bend * mid * np.tanh(2.0 * rg)
        )
        green = y_film - _LUMA[0] * rg2 - _LUMA[2] * bg2
        candidate = np.stack(
            (green + rg2, green, green + bg2), axis=-1
        )

        residual = (candidate - encoded) * amount
        positive = residual > 0.0
        negative = residual < 0.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
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
            raise RuntimeError("bounded opponent response escaped display RGB")
        return output.astype(source.dtype)


__all__ = ["BoundedOpponentFilmResponse"]
