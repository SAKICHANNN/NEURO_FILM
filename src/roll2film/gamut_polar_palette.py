"""Analytic gamut-polar palette operators for bounded RGB looks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


GAMUT_POLAR_PALETTE_SCHEMA = "roll2film.gamut_polar_palette.v1"


def _readonly_vector(
    value: np.ndarray, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    array = array.copy()
    array.setflags(write=False)
    return array


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data with shape (..., 3)")
    return values


def _opponent_basis(luma_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = luma_weights / np.linalg.norm(luma_weights)
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(normal @ reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def _ray_limit(
    luma: np.ndarray, direction: np.ndarray, *, epsilon: float
) -> np.ndarray:
    upper = np.full(direction.shape, np.inf, dtype=np.float64)
    np.divide(
        1.0 - luma[:, None],
        direction,
        out=upper,
        where=direction > epsilon,
    )
    lower = np.full(direction.shape, np.inf, dtype=np.float64)
    np.divide(
        -luma[:, None],
        direction,
        out=lower,
        where=direction < -epsilon,
    )
    limit = np.min(np.minimum(upper, lower), axis=1)
    if np.any(limit < 0.0) or np.any(~np.isfinite(limit)):
        raise RuntimeError("failed to intersect opponent ray with RGB cube")
    return limit


def _bend_forward(value: np.ndarray, bend: np.ndarray | float) -> np.ndarray:
    return value + bend * value * (1.0 - value)


def _bend_inverse(value: np.ndarray, bend: np.ndarray | float) -> np.ndarray:
    bend_values = np.broadcast_to(np.asarray(bend, dtype=np.float64), value.shape)
    small = np.abs(bend_values) < 1e-14
    discriminant = (1.0 + bend_values) ** 2 - 4.0 * bend_values * value
    if np.any(discriminant < -1e-14):
        raise RuntimeError("quadratic bend inverse has negative discriminant")
    denominator = 1.0 + bend_values + np.sqrt(
        np.maximum(discriminant, 0.0)
    )
    solved = np.divide(
        2.0 * value,
        denominator,
        out=value.copy(),
        where=~small,
    )
    solved = np.where(small, value, solved)
    solved = np.where(value == 0.0, 0.0, solved)
    return np.where(value == 1.0, 1.0, solved)


@dataclass(frozen=True)
class GamutPolarPaletteOperator:
    """Tone, hue and chroma diffeomorphisms in exact RGB-cube coordinates."""

    luma_weights: np.ndarray
    tone_bend: float
    shadow_mobius: np.ndarray
    highlight_mobius: np.ndarray
    shadow_hue_rotation: float
    highlight_hue_rotation: float
    chroma_coefficients: np.ndarray
    maximum_mobius_magnitude: float = 0.28
    maximum_absolute_tone_bend: float = 0.55
    maximum_absolute_chroma_bend: float = 0.78
    maximum_absolute_hue_rotation_radians: float = 0.24
    epsilon: float = 1e-14
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        weights = _readonly_vector(self.luma_weights, (3,), "luma_weights")
        if np.any(weights <= 0.0) or abs(float(np.sum(weights)) - 1.0) > 1e-12:
            raise ValueError("luma_weights must be positive and sum to one")
        shadow = _readonly_vector(self.shadow_mobius, (2,), "shadow_mobius")
        highlight = _readonly_vector(
            self.highlight_mobius, (2,), "highlight_mobius"
        )
        chroma = _readonly_vector(
            self.chroma_coefficients, (6,), "chroma_coefficients"
        )
        if not 0.0 < self.maximum_mobius_magnitude < 1.0:
            raise ValueError("maximum_mobius_magnitude must be in (0, 1)")
        if max(np.linalg.norm(shadow), np.linalg.norm(highlight)) > (
            self.maximum_mobius_magnitude + 1e-12
        ):
            raise ValueError("Mobius parameter exceeds the frozen magnitude")
        if (
            not np.isfinite(self.tone_bend)
            or abs(self.tone_bend) > self.maximum_absolute_tone_bend
            or self.maximum_absolute_tone_bend >= 1.0
        ):
            raise ValueError("tone bend exceeds the frozen bound")
        for name, value in (
            ("shadow_hue_rotation", self.shadow_hue_rotation),
            ("highlight_hue_rotation", self.highlight_hue_rotation),
        ):
            if (
                not np.isfinite(value)
                or abs(value) > self.maximum_absolute_hue_rotation_radians
            ):
                raise ValueError(f"{name} exceeds the frozen bound")
        if not 0.0 < self.maximum_absolute_chroma_bend < 1.0:
            raise ValueError("maximum_absolute_chroma_bend must be in (0, 1)")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 requires linear_srgb_d65")
        object.__setattr__(self, "luma_weights", weights)
        object.__setattr__(self, "shadow_mobius", shadow)
        object.__setattr__(self, "highlight_mobius", highlight)
        object.__setattr__(self, "chroma_coefficients", chroma)

    @property
    def opponent_basis(self) -> tuple[np.ndarray, np.ndarray]:
        return _opponent_basis(self.luma_weights)

    def _coordinates(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        luma = rgb @ self.luma_weights
        difference = rgb - luma[:, None]
        first, second = self.opponent_basis
        x = difference @ first
        y = difference @ second
        radius = np.hypot(x, y)
        hue = np.arctan2(y, x)
        direction = (
            np.cos(hue)[:, None] * first
            + np.sin(hue)[:, None] * second
        )
        limit = _ray_limit(luma, direction, epsilon=self.epsilon)
        saturation = np.divide(
            radius,
            limit,
            out=np.zeros_like(radius),
            where=limit > self.epsilon,
        )
        saturation = np.where(
            np.abs(saturation) <= 1e-12, 0.0, saturation
        )
        saturation = np.where(
            np.abs(saturation - 1.0) <= 1e-12, 1.0, saturation
        )
        if np.any(saturation < -1e-11) or np.any(saturation > 1.0 + 1e-11):
            raise RuntimeError("RGB point lies outside analytic gamut coordinates")
        return luma, hue, saturation

    def _hue_parameters(self, luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mobius = (
            (1.0 - luma[:, None]) * self.shadow_mobius
            + luma[:, None] * self.highlight_mobius
        )
        rotation = (
            (1.0 - luma) * self.shadow_hue_rotation
            + luma * self.highlight_hue_rotation
        )
        return mobius[:, 0] + 1j * mobius[:, 1], rotation

    def _hue_forward(self, hue: np.ndarray, luma: np.ndarray) -> np.ndarray:
        z = np.exp(1j * hue)
        parameter, rotation = self._hue_parameters(luma)
        mapped = np.exp(1j * rotation) * (z - parameter) / (
            1.0 - np.conjugate(parameter) * z
        )
        return np.angle(mapped)

    def _hue_inverse(self, hue: np.ndarray, luma: np.ndarray) -> np.ndarray:
        mapped = np.exp(1j * hue)
        parameter, rotation = self._hue_parameters(luma)
        unrotated = mapped * np.exp(-1j * rotation)
        original = (unrotated + parameter) / (
            1.0 + np.conjugate(parameter) * unrotated
        )
        return np.angle(original)

    def _chroma_bend(self, luma: np.ndarray, hue: np.ndarray) -> np.ndarray:
        signed_luma = 2.0 * luma - 1.0
        cosine = np.cos(hue)
        sine = np.sin(hue)
        features = np.column_stack(
            (
                np.ones(len(luma), dtype=np.float64),
                signed_luma,
                cosine,
                sine,
                signed_luma * cosine,
                signed_luma * sine,
            )
        )
        return self.maximum_absolute_chroma_bend * np.tanh(
            features @ self.chroma_coefficients
        )

    def _reconstruct(
        self, luma: np.ndarray, hue: np.ndarray, saturation: np.ndarray
    ) -> np.ndarray:
        first, second = self.opponent_basis
        direction = (
            np.cos(hue)[:, None] * first
            + np.sin(hue)[:, None] * second
        )
        limit = _ray_limit(luma, direction, epsilon=self.epsilon)
        output = luma[:, None] + (saturation * limit)[:, None] * direction
        boundary_rows = np.flatnonzero(saturation == 1.0)
        if len(boundary_rows):
            row_directions = direction[boundary_rows]
            upper = np.full(row_directions.shape, np.inf, dtype=np.float64)
            lower = np.full(row_directions.shape, np.inf, dtype=np.float64)
            np.divide(
                1.0 - luma[boundary_rows, None],
                row_directions,
                out=upper,
                where=row_directions > self.epsilon,
            )
            np.divide(
                -luma[boundary_rows, None],
                row_directions,
                out=lower,
                where=row_directions < -self.epsilon,
            )
            bounds = np.minimum(upper, lower)
            minimum = np.min(bounds, axis=1, keepdims=True)
            active_rows, active_channels = np.nonzero(
                np.abs(bounds - minimum) <= 1e-12
            )
            output[
                boundary_rows[active_rows], active_channels
            ] = np.where(
                row_directions[active_rows, active_channels] > 0.0,
                1.0,
                0.0,
            )
        return output

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        flat = values.reshape(-1, 3)
        luma, hue, saturation = self._coordinates(flat)
        output_luma = _bend_forward(luma, self.tone_bend)
        output_hue = self._hue_forward(hue, luma)
        chroma_bend = self._chroma_bend(luma, hue)
        output_saturation = _bend_forward(saturation, chroma_bend)
        output = self._reconstruct(
            output_luma, output_hue, output_saturation
        )
        tolerance = 2e-12
        if np.any(output < -tolerance) or np.any(output > 1.0 + tolerance):
            raise RuntimeError("gamut-polar operator escaped the RGB cube")
        return output.reshape(shape)

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        flat = values.reshape(-1, 3)
        output_luma, output_hue, output_saturation = self._coordinates(flat)
        luma = _bend_inverse(output_luma, self.tone_bend)
        hue = self._hue_inverse(output_hue, luma)
        chroma_bend = self._chroma_bend(luma, hue)
        saturation = _bend_inverse(output_saturation, chroma_bend)
        restored = self._reconstruct(luma, hue, saturation)
        tolerance = 2e-10
        if np.any(restored < -tolerance) or np.any(restored > 1.0 + tolerance):
            raise RuntimeError("gamut-polar inverse escaped the RGB cube")
        return restored.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GAMUT_POLAR_PALETTE_SCHEMA,
            "working_space": self.working_space,
            "luma_weights": self.luma_weights.tolist(),
            "tone_bend": self.tone_bend,
            "shadow_mobius": self.shadow_mobius.tolist(),
            "highlight_mobius": self.highlight_mobius.tolist(),
            "shadow_hue_rotation": self.shadow_hue_rotation,
            "highlight_hue_rotation": self.highlight_hue_rotation,
            "chroma_coefficients": self.chroma_coefficients.tolist(),
            "maximum_mobius_magnitude": self.maximum_mobius_magnitude,
            "maximum_absolute_tone_bend": self.maximum_absolute_tone_bend,
            "maximum_absolute_chroma_bend": self.maximum_absolute_chroma_bend,
            "maximum_absolute_hue_rotation_radians": (
                self.maximum_absolute_hue_rotation_radians
            ),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GamutPolarPaletteOperator":
        if payload.get("schema") != GAMUT_POLAR_PALETTE_SCHEMA:
            raise ValueError("unsupported gamut-polar palette schema")
        return cls(
            luma_weights=np.asarray(payload["luma_weights"], dtype=np.float64),
            tone_bend=float(payload["tone_bend"]),
            shadow_mobius=np.asarray(payload["shadow_mobius"], dtype=np.float64),
            highlight_mobius=np.asarray(
                payload["highlight_mobius"], dtype=np.float64
            ),
            shadow_hue_rotation=float(payload["shadow_hue_rotation"]),
            highlight_hue_rotation=float(payload["highlight_hue_rotation"]),
            chroma_coefficients=np.asarray(
                payload["chroma_coefficients"], dtype=np.float64
            ),
            maximum_mobius_magnitude=float(
                payload["maximum_mobius_magnitude"]
            ),
            maximum_absolute_tone_bend=float(
                payload["maximum_absolute_tone_bend"]
            ),
            maximum_absolute_chroma_bend=float(
                payload["maximum_absolute_chroma_bend"]
            ),
            maximum_absolute_hue_rotation_radians=float(
                payload["maximum_absolute_hue_rotation_radians"]
            ),
            epsilon=float(payload["epsilon"]),
            working_space=str(payload["working_space"]),
        )


def operator_from_config(
    candidate: dict[str, Any], witness: dict[str, Any]
) -> GamutPolarPaletteOperator:
    return GamutPolarPaletteOperator(
        luma_weights=np.asarray(candidate["luma_weights"], dtype=np.float64),
        tone_bend=float(witness["tone_bend"]),
        shadow_mobius=np.asarray(witness["shadow_mobius"], dtype=np.float64),
        highlight_mobius=np.asarray(
            witness["highlight_mobius"], dtype=np.float64
        ),
        shadow_hue_rotation=float(witness["shadow_hue_rotation"]),
        highlight_hue_rotation=float(witness["highlight_hue_rotation"]),
        chroma_coefficients=np.asarray(
            witness["chroma_coefficients"], dtype=np.float64
        ),
        maximum_mobius_magnitude=float(candidate["maximum_mobius_magnitude"]),
        maximum_absolute_tone_bend=float(
            candidate["maximum_absolute_tone_bend"]
        ),
        maximum_absolute_chroma_bend=float(
            candidate["maximum_absolute_chroma_bend"]
        ),
        maximum_absolute_hue_rotation_radians=float(
            candidate["maximum_absolute_hue_rotation_radians"]
        ),
    )


def finite_difference_jacobians(
    operator: GamutPolarPaletteOperator,
    points: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    values = _validate_rgb(points)
    if (
        values.ndim != 2
        or not np.isfinite(step)
        or step <= 0.0
        or np.any(values < step)
        or np.any(values > 1.0 - step)
    ):
        raise ValueError("points must be finite interior RGB rows for the given step")
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append(
            (operator.apply(values + offset) - operator.apply(values - offset))
            / (2.0 * step)
        )
    return np.stack(columns, axis=-1)


__all__ = [
    "GAMUT_POLAR_PALETTE_SCHEMA",
    "GamutPolarPaletteOperator",
    "finite_difference_jacobians",
    "operator_from_config",
]
