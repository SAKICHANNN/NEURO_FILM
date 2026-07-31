"""Bounded AO6-direction residual over the deterministic safe-rich base."""

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
class FactorizedAO6PerceptualResidual:
    """Add a factorized AO6 residual without re-applying the base direction.

    ``source``, ``base`` and ``candidate`` are fixed renders of the same
    display-sRGB pixels.  The residual is the candidate appearance remaining
    after the base in D65 Lab; its lightness and chroma components are
    independently soft-bounded.  A second, post-gamut Lab bound limits the
    actual increment over the base before analytical RGB rail safety.
    """

    residual_strength: float = 0.6
    lightness_soft_cap: float = 5.0
    chroma_soft_cap: float = 10.0
    maximum_final_delta_e76: float = 8.0
    minimum_lightness: float = 0.1
    maximum_lightness: float = 99.9
    residual_epsilon: float = 1e-12
    gamut_iterations: int = 24
    encoded_margin: float = 1.0 / 65535.0
    row_chunk: int = 128
    numeric_gamut_tolerance: float = 2e-6
    perceptual_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        finite = (
            self.residual_strength,
            self.lightness_soft_cap,
            self.chroma_soft_cap,
            self.maximum_final_delta_e76,
            self.minimum_lightness,
            self.maximum_lightness,
            self.residual_epsilon,
            self.encoded_margin,
            self.numeric_gamut_tolerance,
            self.perceptual_tolerance,
        )
        if (
            not all(np.isfinite(finite))
            or not 0.0 <= self.residual_strength <= 1.0
            or self.lightness_soft_cap <= 0.0
            or self.chroma_soft_cap <= 0.0
            or self.maximum_final_delta_e76 <= 0.0
            or not 0.0 < self.minimum_lightness < self.maximum_lightness
            or self.maximum_lightness >= 100.0
            or self.residual_epsilon <= 0.0
            or self.gamut_iterations <= 0
            or not 0.0 < self.encoded_margin < 0.01
            or self.row_chunk <= 0
            or not 0.0 < self.numeric_gamut_tolerance <= 1e-4
            or not 0.0 < self.perceptual_tolerance <= 0.01
        ):
            raise ValueError("invalid factorized AO6 perceptual residual")

    @staticmethod
    def _lab(encoded: np.ndarray) -> np.ndarray:
        return linear_rgb_to_lab(
            encoded_srgb_to_linear(encoded).astype(np.float32),
            working_space="linear_srgb",
        ).astype(np.float64)

    @staticmethod
    def _soft_bound(
        value: np.ndarray,
        *,
        cap: float,
        epsilon: float,
    ) -> np.ndarray:
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        scale = np.divide(
            cap * np.tanh(norm / cap),
            norm,
            out=np.ones_like(norm),
            where=norm > epsilon,
        )
        return value * scale

    def _factorized_residual(
        self,
        source: np.ndarray,
        base: np.ndarray,
        candidate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_lab = self._lab(base)
        candidate_lab = self._lab(candidate)
        remaining = candidate_lab - base_lab
        lightness = self._soft_bound(
            remaining[..., :1],
            cap=self.lightness_soft_cap,
            epsilon=self.residual_epsilon,
        )
        chroma = self._soft_bound(
            remaining[..., 1:],
            cap=self.chroma_soft_cap,
            epsilon=self.residual_epsilon,
        )
        return base_lab, np.concatenate((lightness, chroma), axis=-1)

    @staticmethod
    def _analytical_scale(
        origin: np.ndarray,
        residual: np.ndarray,
        *,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            upper_scale = np.where(
                residual > 0.0,
                (upper - origin) / residual,
                np.inf,
            )
            lower_scale = np.where(
                residual < 0.0,
                (lower - origin) / residual,
                np.inf,
            )
        return np.clip(
            np.min(np.minimum(upper_scale, lower_scale), axis=-1),
            0.0,
            1.0,
        )

    def _encode_from_lab(
        self,
        target_lab: np.ndarray,
        *,
        base: np.ndarray,
    ) -> np.ndarray:
        candidate_linear = lab_to_linear_rgb(
            target_lab.astype(np.float32),
            working_space="linear_srgb",
        ).astype(np.float64)
        base_linear = encoded_srgb_to_linear(base).astype(np.float64)
        linear_residual = candidate_linear - base_linear
        linear_scale = self._analytical_scale(
            base_linear,
            linear_residual,
            lower=0.0,
            upper=1.0,
        )
        safe_linear = base_linear + linear_residual * linear_scale[..., None]
        excursion = max(
            float(np.max(-safe_linear)),
            float(np.max(safe_linear - 1.0)),
        )
        if excursion > self.numeric_gamut_tolerance:
            raise RuntimeError("analytical linear safety escaped gamut")
        safe_linear = np.minimum(np.maximum(safe_linear, 0.0), 1.0)
        encoded_candidate = linear_srgb_to_encoded(safe_linear)
        encoded_residual = encoded_candidate - base
        encoded_scale = self._analytical_scale(
            base,
            encoded_residual,
            lower=self.encoded_margin,
            upper=1.0 - self.encoded_margin,
        )
        return base + encoded_residual * encoded_scale[..., None]

    def _apply_chunk(
        self,
        source: np.ndarray,
        base: np.ndarray,
        candidate: np.ndarray,
        *,
        strength: float,
    ) -> np.ndarray:
        base_lab, residual = self._factorized_residual(
            source,
            base,
            candidate,
        )
        target_lab = base_lab + strength * residual
        target_lab[..., 0] = np.clip(
            target_lab[..., 0],
            self.minimum_lightness,
            self.maximum_lightness,
        )
        target_lab = compress_chroma_to_working_gamut(
            target_lab.astype(np.float32),
            working_space="linear_srgb",
            iterations=self.gamut_iterations,
            tolerance=self.numeric_gamut_tolerance,
        )

        # Gamut mapping can change the intended Lab vector substantially near
        # extreme RGB corners.  Measure the actual encoded candidate and impose
        # one final bound relative to the safe base.
        first = self._encode_from_lab(target_lab, base=base)
        first_lab = self._lab(first)
        final_delta = first_lab - base_lab
        norm = np.linalg.norm(final_delta, axis=-1, keepdims=True)
        final_scale = np.minimum(
            1.0,
            np.divide(
                self.maximum_final_delta_e76,
                norm,
                out=np.ones_like(norm),
                where=norm > self.residual_epsilon,
            ),
        )
        bounded_lab = base_lab + final_delta * final_scale
        output = self._encode_from_lab(bounded_lab, base=base)
        actual = np.linalg.norm(self._lab(output) - base_lab, axis=-1)
        if float(np.max(actual)) > (
            self.maximum_final_delta_e76 + self.perceptual_tolerance
        ):
            raise RuntimeError("final perceptual residual escaped bound")
        endpoint = np.all(source == 0.0, axis=-1) | np.all(
            source == 1.0,
            axis=-1,
        )
        return np.where(endpoint[..., None], source, output)

    def apply(
        self,
        source_rgb: np.ndarray,
        base_rgb: np.ndarray,
        candidate_rgb: np.ndarray,
        *,
        strength: float | None = None,
    ) -> np.ndarray:
        arrays = tuple(
            np.asarray(value)
            for value in (source_rgb, base_rgb, candidate_rgb)
        )
        source, base, candidate = arrays
        amount = (
            self.residual_strength if strength is None else float(strength)
        )
        if (
            any(
                value.dtype
                not in {np.dtype(np.float32), np.dtype(np.float64)}
                for value in arrays
            )
            or any(value.shape != source.shape for value in arrays[1:])
            or source.ndim != 3
            or source.shape[-1] != 3
            or source.size == 0
            or any(not np.all(np.isfinite(value)) for value in arrays)
            or any(np.any(value < 0.0) or np.any(value > 1.0) for value in arrays)
            or not np.isfinite(amount)
            or not 0.0 <= amount <= 1.0
        ):
            raise ValueError(
                "expected matching finite HxWx3 encoded RGB and strength in [0,1]"
            )
        if amount == 0.0:
            return np.array(base, copy=True)
        output = np.empty_like(base)
        for y0 in range(0, source.shape[0], self.row_chunk):
            y1 = min(y0 + self.row_chunk, source.shape[0])
            output[y0:y1] = self._apply_chunk(
                source[y0:y1].astype(np.float64),
                base[y0:y1].astype(np.float64),
                candidate[y0:y1].astype(np.float64),
                strength=amount,
            ).astype(base.dtype)
        if (
            not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("factorized AO6 perceptual residual escaped RGB")
        return output


__all__ = ["FactorizedAO6PerceptualResidual"]
